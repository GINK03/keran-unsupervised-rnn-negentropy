from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Merge
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.core import Dropout, Reshape, Permute, RepeatVector, Flatten, Lambda
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam, Adagrad, Nadam, SGD, Adadelta, Adamax
from keras.utils.data_utils import get_file
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
import numpy as np
import random
import sys
import glob
from time import gmtime, strftime
import copy
import pickle
import json
import tensorflow as tf
import math

def log_softmax(x):
  xdev = x - x.max(1, keepdims=True)
  return xdev - K.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def categorical_crossentropy_logdomain(log_predictions, targets):
  return -T.sum(targets * log_predictions, axis=1)

def free_random_fold():
  neg_ent = {}
  LEN = 21
  feat_vec = json.loads(open('./fastText_vector.json', 'r').read())
  txt = list(open(TARGET_FNAME).read().replace('\n', '').replace(' ', ''))
  for ti in range(0, len(txt) - LEN, 1):
     crop   = list(copy.copy(txt[ti: ti+LEN]))
     crop[LEN//2] = '☆'
     for ci, _ in enumerate(crop): 
       if random.random() <= 0.25: 
         crop[ci] = '☆'
     origin = copy.copy(txt[ti: ti+LEN])
     for o in origin:
       if feat_vec.get(o) is None:
         continue
     source = ''.join(crop)
     origin = ''.join(origin);
     neg_ent[origin] = source
  open(TARGET_DIR + 'neg_ent.json', 'w').write(json.dumps(neg_ent))

def build_model(maxlen=None, feats=None):
  print('Build model...')
  situations  = 4
  vector_size = 256
  model_text = Sequential()
  model_text.add(Reshape( (maxlen, vector_size, ), input_shape=(maxlen, vector_size)))
  model_text.add(GRU(int(128*8)))
  model_text.add(BN(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None) )
  model_text.add(Dropout(0.5))
  final_model = Sequential()
  final_model.add(model_text)
  final_model.add(Dense(len(feats)))
  final_model.add(Activation('softmax') )
  optimizer = Adam()
  final_model.compile(loss='categorical_crossentropy', optimizer=optimizer)
  return final_model

def sample(preds, temperature):
    preds_raw = np.asarray(preds).astype('float64')
    preds = np.log(preds_raw) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return (np.argmax(probas), np.var(preds_raw))

def dynast(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return (np.argmax(preds), np.var(preds))

def monoid(preds, y_char, feat_indices, termperture=1.0):
  preds = np.asarray(preds).astype('float64')
  argmax = np.argmax(preds)
  try:
    id = feat_indices[y_char]
  except KeyError as e:
    id = 0
  return  preds[id], argmax, np.var(preds)

def train(file_name, size):
  feat_vec = json.loads(open('./fastText_vector.json', 'r').read())
  neg_ent  = json.loads(open(TARGET_DIR + 'neg_ent.json', 'r').read())
  datasets = open(TARGET_FNAME).read()
  feats = sorted(list(set(datasets)))
  print('total feats:', len(feats))
  feat_indices = dict((f, i) for i, f in enumerate(feats))
  indices_feat = dict((i, f) for i, f in enumerate(feats))
  open(TARGET_DIR + 'feat_indices.pkl', 'wb').write(pickle.dumps(feat_indices))
  open(TARGET_DIR + 'indices_feat.pkl', 'wb').write(pickle.dumps(indices_feat))
  maxlen = len(list(neg_ent.values())[0])
  step = 1
  vector_size = 256
  sentences = []
  next_feats = []
  
  buff = []
  alldata = [ (t,s) for t,s in neg_ent.items() ]
  inv_neg_ent = { s:t for t,s in neg_ent.items() }
  train  = alldata[:(len(alldata)//10)*8]
  eval   = alldata[(len(alldata)//10)*8+1:]
  open(TARGET_DIR + 'to_eval.json', 'w').write(json.dumps(eval) )
  for target, source in train:
    buff.append( (0, source, target[maxlen//2] ) )
      
  random_crop = buff
  random.shuffle(random_crop)
  for ri, sentence in enumerate(random_crop):
    sentences.append( sentence )
    if ri%500 == 0:
      print("enum", sentence[0], ''.join(sentence[1]), sentence[2])
 
  eval_sentences = []
  for target, source in eval:
    eval_sentences.append( (0, source, target[maxlen//2]) )

  print('nb sequences:', len(sentences))
  print('Vectorization...')
  X1 = np.zeros((len(sentences), maxlen, vector_size), dtype="float")
  y  = np.zeros((len(sentences), len(feats)), dtype=np.bool)
  # 実験的
  random.shuffle(sentences)
  len_sentences = len(sentences)
  for i, sentence in enumerate(sentences):
    enum  = sentence[0]
    x_sen = sentence[1]
    y_sen = sentence[2]
    if i % 5000 == 0:
      print('progress %d/%d'%(i, len_sentences))
    for t, feat in enumerate(x_sen):
      try:
        X1[i, t, :vector_size] = np.array( feat_vec[feat] ) 
      except KeyError as e:
        continue
    y[i, feat_indices[y_sen]] = 1
  print('Building model...')
  model = build_model(maxlen=maxlen, feats=feats)
  
  print('Starting Learning...')
  for iteration in range(1, 10000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit([X1], y, batch_size=128, nb_epoch=1)
    MODEL_NAME     = TARGET_DIR + "%s.%09d.model"%("combine", iteration)
    model.save(MODEL_NAME)
    if iteration%1 == 0:
      if iteration == 80*2 : sys.exit()
      for _ in range(3):
        for type, evaluater, source_set in [("dynast", dynast, sentences), ("dynast(eval)", dynast, eval_sentences)]:
          diversity = 1.2
          print()
          print('----- diversity:', diversity)
          generated = ''
          start_index = np.random.randint(0, len(source_set) - maxlen)
          sentence = source_set[start_index]
          seed_enum  = sentence[0]
          x_sen_     = sentence[1]
          y_sen_     = sentence[2]
          generated += ''.join(x_sen_)
          print('----- Generating with seed: <%d> type <%s>"'%(seed_enum,type) , ''.join(x_sen_) + '"')
          print('----- Target Text: %s'%( str(inv_neg_ent.get(x_sen_)) ) )
          sys.stdout.write(generated + "|")
          x1_ = np.zeros((1, maxlen, vector_size))
          for t, feat in enumerate(x_sen_):
            try:
              x1_[0, t, :vector_size]         = np.array(feat_vec[feat])
            except:
              continue
          preds = model.predict([x1_], verbose=0)[0]
          next_index, var = evaluater(preds, diversity)
          next_feat = indices_feat[next_index]
          generated += next_feat
          x_sen_ = x_sen_[1:] + next_feat
          sys.stdout.write(next_feat)
          sys.stdout.flush()
          print()

def eval():
  feat_vec = json.loads(open('./fastText_vector.json', 'r').read())
  feat_indices = pickle.loads(open(TARGET_DIR + 'feat_indices.pkl', 'rb').read())
  indices_feat = pickle.loads(open(TARGET_DIR + 'indices_feat.pkl', 'rb').read())
  MODEL_NAME = sorted(glob.glob(TARGET_DIR + '*.model'))[-1]
  print('using model name is %s'%MODEL_NAME)
  maxlen = 21
  step = 1
  vector_size = 256
  situations = 4
  model = load_model(MODEL_NAME)
  print('Starting Learning...')
  ths = []
  for _ in range(100, 1000, 100):
    ths.append( -0.01*_ )
  import testcast
  #eval_target = testcast.eval_target[100:200]
  eval_target = list(testcast.midway)
  for ti in range(11, len(eval_target) - 11, 1):
    if random.random() < 0.30:
      eval_target[ti] = '☆'
    pass
  eval_target = ''.join(eval_target)
  f = open(TARGET_DIR + 'revocer.txt', 'w')
  print(eval_target)
  f.write(eval_target + '\n')
  o = []
  io= []
  result = ""
  for _ in range(eval_target.count('☆')):
    print( "loop", _ )
    prob_set = []
    for ei in range(11, len(eval_target)-11, 1):
      for type, evaluater in [("dynast", dynast)]:
        diversity = 1.2
        generated = ''
        seed_enum  = 1
        enum       = 1
        x1_ = np.zeros((1, maxlen, vector_size))
        x_sen_ = eval_target[ei-10:ei+11]
        y_sen_ = testcast.bible[ei]
        if eval_target[ei] != '☆':
          continue
        for t, feat in enumerate(x_sen_):
          try:
            x1_[0, t, :vector_size]         = np.array(feat_vec[feat])
          except KeyError as e:
            pass
        preds = model.predict([x1_], verbose=0)[0]
        prob, argmax, var = monoid(preds, y_sen_, feat_indices, diversity)
        try:
          prob = math.log(prob)
        except ValueError as e:
          prob = prob
        prob_set.append( (ei, prob, indices_feat[argmax]) )
    id, prob, argmax = max(prob_set, key=lambda x:x[1])
    eval_target     = list(eval_target)
    eval_target[id] = argmax
    eval_target     = ''.join(eval_target)
    print(id, prob, argmax)
    print(eval_target[id - 11: id + 11])
    f.write(eval_target + '\n')
  print()

TARGET_FNAME = './midway/midway.txt'
TARGET_DIR   = './midway/'
#TARGET_FNAME = './bible/midway.txt'
#TARGET_DIR   = './bible/'
def main():
  if '--train' in sys.argv:
     train(file_name=TARGET_FNAME, size=1)
  if '--eval' in sys.argv:
     eval()
  if '--freefold' in sys.argv:
     free_random_fold()     
if __name__ == '__main__':
  main()
