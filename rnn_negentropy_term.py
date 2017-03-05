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
  neg_ent = []
  LEN = 21
  feat_vec = json.loads(open('./anime/term_vec.json', 'r').read())
  txt = list(open('./anime/anime_news.txt').read().replace('\n', ' ').split())
  for ti in range(0, len(txt) - LEN, 1):
     crop   = list(copy.copy(txt[ti: ti+LEN]))
     crop[LEN//2] = '※'
     for ci, _ in enumerate(crop): 
       if random.random() <= 0.25: 
         crop[ci] = '※'
         pass
     origin = copy.copy(txt[ti: ti+LEN])
     for o in origin:
       if feat_vec.get(o) is None:
         continue
     source = crop
     origin = origin;
     neg_ent.append( (origin, source) )
     #print( (source, origin) )
  open('./anime/neg_ent.json', 'w').write(json.dumps(neg_ent))

def build_model(maxlen=None, feats=None):
  print('Build model...')
  situations  = 4
  vector_size = 256
  model_text = Sequential()
  model_text.add(Reshape( (maxlen, vector_size, ), input_shape=(maxlen, vector_size)))
  model_text.add(GRU(int(128*10)))
  model_text.add(BN(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None) )
  model_text.add(Dropout(0.5))
  final_model = Sequential()
  final_model.add(model_text)
  final_model.add(Dense(len(feats)))
  final_model.add(Activation('linear') )
  optimizer = Adam()
  #reduce_lr = ReduceLROnPlateau(monitor='categorical_crossentropy', factor=0.2, patience=5, min_lr=0.0001)
  #final_model.compile(loss='poisson', optimizer=optimizer)
  final_model.compile(loss='mean_squared_error', optimizer=optimizer)
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
  #exp_preds = np.exp(preds)
  #preds = exp_preds / np.sum(exp_preds)
  id = feat_indices[y_char]
  return  preds[id], np.var(np.log(preds))

def train(file_name, size):
  feat_vec = json.loads(open('./anime/term_vec.json', 'r').read())
  neg_ent  = json.loads(open('./anime/neg_ent.json', 'r').read())
  datasets = open('./anime/anime_news.txt').read().replace('\n', ' ').split()
  feats = sorted(list(set(datasets)))
  print('total feats:', len(feats))
  feat_indices = dict((f, i) for i, f in enumerate(feats))
  indices_feat = dict((i, f) for i, f in enumerate(feats))
  open('feat_indices.pkl', 'wb').write(pickle.dumps(feat_indices))
  open('indices_feat.pkl', 'wb').write(pickle.dumps(indices_feat))
  maxlen = len(list(neg_ent)[0][0])
  step = 1
  vector_size = 256
  sentences = []
  next_feats = []
  
  buff = []
  alldata = [ (t,s) for t,s in neg_ent ]
  inv_neg_ent = { ''.join(s):t for t,s in neg_ent }
  train  = alldata[:(len(alldata)//10)*1]
  eval   = alldata[(len(alldata)//10)*8+1:]
  open('to_eval.json', 'w').write(json.dumps(eval) )
  for target, source in train:
    #print(target, source)
    buff.append( (0, source, target[maxlen//2] ) )
    #print( buff[-1] )
      
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
    #print(y_sen)
    if i % 5000 == 0:
      print('progress %d/%d'%(i, len_sentences))
    for t, feat in enumerate(x_sen):
      #print(t,feat)
      #print(feat_vec[feat])
      try:
        X1[i, t, :vector_size] = np.array( feat_vec[feat] ) 
      except KeyError as e:
        print("keyerror", feat)
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
    MODEL_NAME     = "./hitoribocchi/%s.%09d.model"%("combine", iteration)
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
          print('----- Target Text: %s'%( str(inv_neg_ent.get(''.join(x_sen_))) ) )
          sys.stdout.write(generated + "|")
          #for i in range(100):
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
          x_sen_ = x_sen_[1:]
          x_sen_.append( next_feat )
          sys.stdout.write(next_feat)
          #sys.stdout.write(str(var)[:3])
          sys.stdout.flush()
          print()

def eval():
  feat_vec = json.loads(open('./fastText_vector.json', 'r').read())
  feat_indices = pickle.loads(open('./feat_indices.pkl', 'rb').read())
  indices_feat = pickle.loads(open('./indices_feat.pkl', 'rb').read())
  MODEL_NAME = sorted(glob.glob('./models/*'))[-1]
  print('using model name is %s'%MODEL_NAME)
  maxlen = 11
  step = 1
  vector_size = 256
  situations = 4
  model = load_model(MODEL_NAME)
  print('Starting Learning...')
  ths = []
  for _ in range(100, 1000, 100):
    ths.append( -0.01*_ )
  for th in ths:
    print("th %f"%th)
    import testcast
    #eval_target = testcast.eval_target[100:200]
    eval_target = testcast.sec
    o = []
    io= []
    result = ""
    for slicer in range(0, len(eval_target) - maxlen - maxlen, 1):
      tmp = list(copy.copy(eval_target[slicer: slicer+maxlen]))
      tmp[int(maxlen/2)] = 'T'
      x_sen_     =  ''.join(tmp)
      y_sen_     = eval_target[slicer+int(maxlen/2)]
      #print( y_sen_, x_sen_inv_ )
      for type, evaluater in [("dynast", dynast)]:#[("dynast", dynast), ("sample", sample)]:
        diversity = 1.2
        generated = ''
        seed_enum  = 1
        enum       = 1
        #sys.stdout.write(generated)
        x1_ = np.zeros((1, maxlen, vector_size))
        for t, feat in enumerate(x_sen_):
          x1_[0, t, :vector_size]         = np.array(feat_vec[feat])
        preds = model.predict([x1_], verbose=0)[0]
        try:
          prob, var = monoid(preds, y_sen_, feat_indices, diversity)
          prob = math.log(prob)
        except:
          prob, var = 0, 0
        sys.stdout.write("   " + x_sen_)
        sys.stdout.write("\n")
        sys.stdout.write(y_sen_ + "prob" + str(prob) + " var" + str(var))
        sys.stdout.write("\n")
        sys.stdout.flush()
        
    print()

def main():
  if '--train' in sys.argv:
     #FILE_NAME = './eva/vm.txt'
     FILE_NAME = './source/mushokutensei.txt'
     train(file_name=FILE_NAME, size=1)
  if '--eval' in sys.argv:
     eval()
  if '--freefold' in sys.argv:
     free_random_fold()     
if __name__ == '__main__':
  main()
