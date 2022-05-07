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

def pre_trainer():
  frame_size   = 128
  path = "/home/oshanjayawardanav100/biometrics-self-supervised/gait_dataset/idnet/"
  
  users_2 = list(range(17,51)) #Users for dataset 2
  users_1 = list(range(1,17)) #Users for dataset 1
  
  x_train, y_train, x_val, y_val, x_test, y_test, sessions = data_loader_gait(path, classes=users_1, frame_size=frame_size)
  
  classes, counts  = np.unique(y_train, return_counts=True)
  num_classes = len(classes)
  print("minimum samples per user : ", min(counts)) #60
  
  x_train, x_val, x_test = norma(x_train, x_val, x_test)
  print("x_train", x_train.shape)
  print("x_val", x_val.shape)
  print("x_test", x_test.shape)
  
  #x_train, y_train = user_data_split(x_train ,y_train , samples_per_user=samples_per_user)
  #print("limited training samples : ", x_train.shape[0])
  #classes, counts  = np.unique(y_train, return_counts=True)
  #print(counts)
  
  ks = 3
  con =1
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
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.001, decay_rate=0.95, decay_steps=1000)# 0.0001, 0.9, 100000
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  #optimizer = tf.keras.optimizers.Adam()
  resnettssd.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
  history = resnettssd.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=callback, batch_size=32)
  
  results = resnettssd.evaluate(x_test,y_test)
  test_acc = results[1]
  print("Pre training_test acc:", results[1])
  
  #Calculating kappa score
  metric = tfa.metrics.CohenKappa(num_classes=num_classes, sparse_labels=True)
  metric.update_state(y_true=y_test , y_pred=resnettssd.predict(x_test))
  result = metric.result()
  kappa_score = result.numpy()
  print('"Pre training_kappa score: ',result.numpy())
  
  resnettssd = tf.keras.Model(
            resnettssd.input, resnettssd.layers[-5].output
        )
  resnettssd.summary()
  x_train, y_train, sessions_train = data_loader_gait_pre(path, classes=users_1, frame_size=128)
  enc_results = resnettssd(x_train)
  enc_results = np.array(enc_results)
  X_embedded = TSNE(n_components=2).fit_transform(enc_results)
  fig4 = plt.figure(figsize=(18,12))
  plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_train)
  plt.savefig('graphs/latentspace_scen_1.png')
  plt.close(fig4)
  
  x_train, y_train, sessions_train = data_loader_gait_pre(path, classes=users_2, frame_size=128)
  enc_results = resnettssd(x_train)
  enc_results = np.array(enc_results)
  X_embedded = TSNE(n_components=2).fit_transform(enc_results)
  fig5 = plt.figure(figsize=(18,12))
  plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_train)
  plt.savefig('graphs/latentspace_scen_3.png')
  plt.close(fig5)
  
  return resnettssd