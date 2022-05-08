import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten
from sklearn.manifold import TSNE

from backbones import *
from data_loader import *
from trunks import *

def pre_trainer(config_num):
  
  EPOCHS = 50
  BATCH_SIZE = 32
  
  frame_size   = 30
  path = "/home/oshanjayawardanav100/biometrics-self-supervised/musicid_dataset/"
  
  users_2 = list(range(7,21)) #Users for dataset 2
  users_1 = list(range(1,7)) #Users for dataset 1
  folder_train = ["TrainingSet","TestingSet_secret", "TestingSet"]
  
  x_train, y_train, sessions_train = data_load_origin(path, users=users_1, folders=folder_train, frame_size=30)
  print("training samples : ", x_train.shape[0])
  
  x_train = norma_pre(x_train)
  print("x_train", x_train.shape)
  
  #################################################### Augmentations ########################################################################
  transformations = np.array([DA_Jitter, DA_Scaling, DA_MagWarp, DA_RandSampling, DA_Flip, DA_Drop, DA_TimeWarp, DA_Negation, DA_ChannelShuffle, DA_Permutation])
  con = 2
  sigma_l = np.array([0.1*con, 0.2*con, 0.2*con, None, None, 3, 0.2*con, None, None, 0.1*con])
  
  x_train, y_train = aug_data(x_train, y_train, transformations, sigma_l, ext=False)
  ###########################################################################################################################################
  
  trunk_lst = [trunk_1, trunk_2, trunk_3, trunk_4, trunk_5, trunk_6]
    
  inputs = []
  for i in range(len(transformations)):
    name = 'input_'+str(i+1)
    inputs.append(Input(shape=(frame_size,x_train.shape[-1]), name=name))
  
  trunk=trunk_lst[config_num]
  trunk=trunk(frame_size, ft_len=x_train.shape[-1])
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
  
  lr_decayed_fn = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.00003, decay_rate=0.95, decay_steps=1000)# 0.0001, 0.9, 100000
  opt = tf.keras.optimizers.Adam(lr_decayed_fn)
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
    history=model.fit(x_train[i], y_train[i], epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE)
  else:
    history=model.fit(x_, y_, epochs=EPOCHS, shuffle=True, callbacks=[Logger()], verbose=False, batch_size=BATCH_SIZE)
  
  fet_extrct=model.layers[len(transformations)]
  
  return fet_extrct