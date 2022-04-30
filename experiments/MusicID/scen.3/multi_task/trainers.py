import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve

from backbones import *
from data_loader import *

def trainer(samples_per_user):
  frame_size   = 30
  path = "/home/oshanjayawardanav100/biometrics-self-supervised/musicid_dataset/"
  
  users_2 = list(range(9,21)) #Users for dataset 1
  users_1 = users = list(range(1,7)) #Users for dataset 2
  folder_train = ["TrainingSet","TestingSet_secret", "TestingSet"]
  
  x_train, y_train, sessions_train = data_load_origin(path, users=users_1, folders=folder_train, frame_size=30)
  print("training samples : ", x_train.shape[0])
  
  x_train = norma_pre(x_train)
  print("x_train", x_train.shape)
  
  transformations=np.array([DA_Jitter, DA_Scaling, DA_MagWarp, DA_TimeWarp,DA_RandSampling,DA_Flip,DA_Drop])
  sigma_l=np.array([1, 2, 2, 2, None, None, None])
  x_train, y_train = aug_data(x_train, y_train, transformations, sigma_l)
  
  con=1
  ks=3
  def trunk():
    input_ = Input(shape=(frame_size,x_train.shape[-1]), name='input_')
    x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same')(input_) 
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(rate=0.1)(x)
    x = resnetblock(x, KS=ks, CR=64*con)
    x = resnetblock(x, KS=ks, CR=256*con)
    x = resnetblock_final(x, CR=1024*con, KS=ks)
    return tf.keras.models.Model(input_,x,name='trunk_')
    
  input_1 = Input(shape=(frame_size,x_train.shape[-1]), name='input_1')
  input_2 = Input(shape=(frame_size,x_train.shape[-1]), name='input_2')
  input_3 = Input(shape=(frame_size,x_train.shape[-1]), name='input_3')
  input_4 = Input(shape=(frame_size,x_train.shape[-1]), name='input_4')
  input_5 = Input(shape=(frame_size,x_train.shape[-1]), name='input_5')
  input_6 = Input(shape=(frame_size,x_train.shape[-1]), name='input_6')
  input_7 = Input(shape=(frame_size,x_train.shape[-1]), name='input_7')
  
  trunk=trunk()
  
  fet_1 = trunk(input_1)
  fet_2 = trunk(input_2)
  fet_3 = trunk(input_3)
  fet_4 = trunk(input_4)
  fet_5 = trunk(input_5)
  fet_6 = trunk(input_6)
  fet_7 = trunk(input_7)
  
  dens_1 = Dense(256, activation='relu', name='dens_1')(fet_1)
  head_1 = Dense(1, activation='sigmoid', name='head_1')(dens_1)
  
  dens_2 = Dense(256, activation='relu', name='dens_2')(fet_2)
  head_2 = Dense(1, activation='sigmoid', name='head_2')(dens_2)
  
  dens_3 = Dense(256, activation='relu', name='dens_3')(fet_3)
  head_3 = Dense(1, activation='sigmoid', name='head_3')(dens_3)
  
  dens_4 = Dense(256, activation='relu', name='dens_4')(fet_4)
  head_4 = Dense(1, activation='sigmoid', name='head_4')(dens_4)
  
  dens_5 = Dense(256, activation='relu', name='dens_5')(fet_5)
  head_5 = Dense(1, activation='sigmoid', name='head_5')(dens_5)
  
  dens_6 = Dense(256, activation='relu', name='dens_6')(fet_6)
  head_6 = Dense(1, activation='sigmoid', name='head_6')(dens_6)
  
  dens_7 = Dense(256, activation='relu', name='dens_7')(fet_7)
  head_7 = Dense(1, activation='sigmoid', name='head_7')(dens_7)
  
  model = tf.keras.models.Model([input_1,input_2,input_3,input_4,input_5,input_6,input_7], [head_1, head_2, head_3, head_4, head_5, head_6, head_7], name='multi-task_self-supervised')
  
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
          print("val_accuracy",val_acc)
  
  callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.1,patience=5,restore_best_weights=True )
  x_=[]
  y_=[]
  for i in range(len(transformations)):
      x_.append(x_train[i])
      y_.append(y_train[i])
  
  history=model.fit(x_, y_, epochs=30 , validation_split=0.2, shuffle=True, callbacks=[Logger()], verbose=False)
  
  ######################################################Transfering##########################################################################################
  
  folder_train = ["TrainingSet"]
  folder_val = ["TestingSet"]
  folder_test = ["TestingSet_secret"]
  
  x_train, y_train, sessions_train = data_load_origin(path, users=users_2, folders=folder_train, frame_size=30)
  print("training samples : ", x_train.shape[0])
  
  x_val, y_val, sessions_val = data_load_origin(path, users=users_2, folders=folder_val, frame_size=30)
  print("validation samples : ", x_val.shape[0])
  
  x_test, y_test, sessions_test = data_load_origin(path, users=users_2, folders=folder_test, frame_size=30)
  print("testing samples : ", x_test.shape[0])
  
  classes, counts  = np.unique(y_train, return_counts=True)
  num_classes = len(classes)
  print("minimum samples per user : ", min(counts)) #60
  
  x_train, x_val, x_test = norma(x_train, x_val, x_test)
  print("x_train", x_train.shape)
  print("x_val", x_val.shape)
  print("x_test", x_test.shape)
  
  x_train, y_train = user_data_split(x_train ,y_train , samples_per_user=samples_per_user)
  print("limited training samples : ", x_train.shape[0])
  classes, counts  = np.unique(y_train, return_counts=True)
  print(counts)
  
  fet_extrct=model.layers[len(transformations)]

  fet_extrct.trainable=False
  inputs = Input(shape=(frame_size, x_train.shape[-1]))
  x = fet_extrct(inputs, training=False)
  x = Dense(256, activation='relu')(x)
  x = Dense(64, activation='relu')(x)
  outputs = Dense(num_classes, activation='softmax')(x)
  resnettssd = Model(inputs, outputs)
  #resnettssd.summary()
  
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=5)
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.001, decay_rate=0.95, decay_steps=1000)# 0.0001, 0.9, 100000
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  #optimizer = tf.keras.optimizers.Adam()
  resnettssd.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
  history = resnettssd.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=callback, batch_size=8)
  
  results = resnettssd.evaluate(x_test,y_test)
  test_acc = results[1]
  print("test acc:", results[1])
  
  #Calculating kappa score
  metric = tfa.metrics.CohenKappa(num_classes=num_classes, sparse_labels=True)
  metric.update_state(y_true=y_test , y_pred=resnettssd.predict(x_test))
  result = metric.result()
  kappa_score = result.numpy()
  print('kappa score: ',result.numpy())
  
  return test_acc, kappa_score