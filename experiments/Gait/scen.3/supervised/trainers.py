import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from backbones import *
from data_loader import *

def trainer(samples_per_user):
  frame_size   = 128
  path = "/home/oshanjayawardanav100/biometrics-self-supervised/gait_dataset/idnet/"
  #path = "/home/oshanjayawardanav100/biometrics-self-supervised/gait_dataset/IDNet's dataset/user_coordinates/"
  
  users_2 = list(range(19,51)) #Users for dataset 2
  users_1 = list(range(1,17)) #Users for dataset 1
  
  x_train, y_train, x_val, y_val, x_test, y_test, sessions = data_loader_gait(path, classes=users_2+users_1, frame_size=frame_size)
  #x_sample, y_sample,sessions_sample = data_loader_gait_pre(path, classes=[17,18], frame_size=128)
  #_, scaler = norma_origin(x_sample)
  
  classes, counts  = np.unique(y_train, return_counts=True)
  print(classes)
  num_classes = len(classes)
  print("num_classes ",num_classes)
  print("minimum samples per user : ", min(counts)) #60
  
  x_train, x_val, x_test = norma(x_train, x_val, x_test)
  #x_train, _ = norma_origin(x_train, scaler)
  #x_val, _ = norma_origin(x_val, scaler)
  #x_test, _ = norma_origin(x_test, scaler)
  print("x_train", x_train.shape)
  print("x_val", x_val.shape)
  print("x_test", x_test.shape)
  
  x_train, y_train = user_data_split(x_train ,y_train , samples_per_user=samples_per_user)
  print("limited training samples : ", x_train.shape[0])
  classes, counts  = np.unique(y_train, return_counts=True)
  print(counts)
  
  ks = 3
  con = 1
  inputs = Input(shape=(frame_size, x_train.shape[-1]))
  x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same')(inputs) 
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = MaxPooling1D(pool_size=4, strides=4)(x)
  x = Dropout(rate=0.1)(x)
  x = resnetblock_final(x, CR=32*con, KS=ks)
  x = Flatten()(x)
  x = Dense(256, activation='relu')(x)
  x = Dense(64, activation='relu')(x)
  outputs = Dense(num_classes, activation='softmax')(x)
  resnettssd = Model(inputs, outputs)
  #resnettssd.summary()
  
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=5)
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.001, decay_rate=0.95, decay_steps=1000000)# 0.0001, 0.9, 100000
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  #optimizer = tf.keras.optimizers.Adam()
  resnettssd.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
  history = resnettssd.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=callback, batch_size=32)
  
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