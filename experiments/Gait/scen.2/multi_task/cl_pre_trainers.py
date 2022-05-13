import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

from backbones import *
from data_loader import *

def cl_pre_trainer(samples_per_user, fet_extrct):
  frame_size   = 128
  path = "/home/oshanjayawardanav100/biometrics-self-supervised/gait_dataset/idnet/"
  
  users_2 = [14, 15, 16, 17, 20, 21, 22, 23, 32, 36, 37, 38, 39, 41, 43, 44, 45, 46, 47, 48, 49, 50] #Users for dataset 2
  users_1 = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] #Users for dataset 1
  
  x_train, y_train, x_val, y_val, x_test, y_test, sessions = data_loader_gait(path, classes=users_1, frame_size=frame_size)
  
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
  
  inputs = Input(shape=(frame_size, x_train.shape[-1]))
  x = fet_extrct(inputs, training=False)
  x = Dense(256, activation='relu')(x)
  x = Dense(64, activation='relu')(x)
  outputs = Dense(num_classes, activation='softmax')(x)
  resnettssd = Model(inputs, outputs)
  #resnettssd.summary()
  
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=5)
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.001/50, decay_rate=0.95, decay_steps=1000)# 0.0001, 0.9, 100000
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
  
  resnettssd = resnettssd.layers[1]
  
  #resnettssd.summary()
  
  return test_acc, kappa_score, resnettssd