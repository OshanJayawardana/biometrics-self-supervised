import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from sklearn.manifold import TSNE

from backbones import *
from data_loader import *

def pre_trainer(transformations, sigma_l, name_aug):
  frame_size   = 30
  path = "/home/oshanjayawardanav100/biometrics-self-supervised/musicid_dataset/"
  
  users_2 = list(range(7,21)) #Users for dataset 2
  users_1 = list(range(1,7)) #Users for dataset 1
  folder_train = ["TrainingSet","TestingSet_secret", "TestingSet"]
  
  x_train, y_train, sessions_train = data_load_origin(path, users=users_1, folders=folder_train, frame_size=30)
  print("training samples : ", x_train.shape[0])
  
  x_train = norma_pre(x_train)
  print("x_train", x_train.shape)
  
  #sigma_l=np.array([0.2, None])
  x_train, y_train = aug_data(x_train, y_train, transformations, sigma_l, ext=False)
  
  con=3
  ks=3
  def trunk():
    input_ = Input(shape=(frame_size,x_train.shape[-1]), name='input_')
    x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same')(input_) 
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(rate=0.1)(x)
    x = resnetblock_final(x, CR=32*con, KS=ks)
    return tf.keras.models.Model(input_,x,name='trunk_')
    
  inputs = []
  for i in range(len(transformations)):
    name = 'input_'+str(i+1)
    inputs.append(Input(shape=(frame_size,x_train.shape[-1]), name=name))
  
  trunk=trunk()
  trunk.summary()
  
  fets = []
  for input_ in inputs:
    fets.append(trunk(input_))
  
  heads = []
  for i, fet in enumerate(fets):
    dens_name = 'dens_'+str(i+1)
    densi_name = 'densi_'+str(i+1)
    head_name = 'head_'+str(i+1)
    dens = Dense(256, activation='relu', name=dens_name)(fet)
    dens = Dense(64, activation='relu', name=densi_name)(dens)
    head = Dense(1, activation='sigmoid', name=head_name)(dens)
    heads.append(head)
  
  if len(transformations)==1:
    model = tf.keras.models.Model(inputs[0], heads[0], name='multi-task_self-supervised')
  else:
    model = tf.keras.models.Model(inputs, heads, name='multi-task_self-supervised')
  
  loss=[]
  loss_weights=[]
  for i in range(len(transformations)):
    loss.append('binary_crossentropy')
    loss_weights.append(1/len(transformations))
  #loss_weights=[1,0.1,0.1,0.1,1,1,0.1]
  
  opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(
      loss=loss,
      loss_weights=loss_weights,
      optimizer=opt,
      metrics=['accuracy']
  )
  
  model.summary()
  
  class Logger(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
          acc=[]
          val_acc=[]
          for i in range(len(transformations)):
              acc.append(logs.get('head_'+str(i+1)+'_accuracy'))
              val_acc.append(logs.get('val_head_'+str(i+1)+'_accuracy'))
          print('='*30,epoch+1,'='*30)
          print('accuracy',acc)
          #print("val_accuracy",val_acc)
  
  callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.1,patience=5,restore_best_weights=True )
  x_=[]
  y_=[]
  for i in range(len(transformations)):
      x_.append(x_train[i])
      y_.append(y_train[i])
  
  if len(transformations)==1:
    history=model.fit(x_train[i], y_train[i], epochs=30, shuffle=True)
  else:
    history=model.fit(x_, y_, epochs=30, shuffle=True, callbacks=[Logger()], verbose=False)
  
  fet_extrct=model.layers[len(transformations)]
  
  x_train, y_train, sessions_train = data_load_origin(path, users=users_2, folders=folder_train, frame_size=30)
  x_train = norma_pre(x_train)
  enc_results = fet_extrct(x_train)
  enc_results = np.array(enc_results)
  X_embedded = TSNE(n_components=2).fit_transform(enc_results)
  fig4 = plt.figure(figsize=(18,12))
  plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_train)
  plt.title(name_aug)
  plt.savefig('graphs/latentspace_scen_3_'+name_aug+'.png')
  plt.close(fig4)
  
  return fet_extrct