import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers
from sklearn.manifold import TSNE
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from projectors import *
from predictors import *
from simsiam import *

from backbones import *
from data_loader import *
from transformations import *

def pre_trainer(pred_config_num):
  
  transformations1, transformations2 = [DA_MagWarp, DA_Jitter], [DA_Scaling, DA_Flip]
  config_num = 2
  reg_con = 0.01
  BATCH_SIZE = 40
  origin = False
  EPOCHS = 50 #50
  frame_size   = 128
  path = "/home/oshanjayawardanav100/biometrics-self-supervised/gait_dataset/idnet/"
  
  users_2 = list(range(17,51)) #Users for dataset 2
  users_1 = list(range(1,17)) #Users for dataset 1
  
  x_train, y_train, x_val, y_val, x_test, y_test, sessions = data_loader_gait(path, classes=users_1, frame_size=frame_size)
  print("training samples : ", x_train.shape[0])
  num_sample=x_train.shape[0]
  
  x_train = norma_pre(x_train)
  print("x_train", x_train.shape)
  
  def aug1_numpy(x): #Jitter 0.5 scale 1
    # x will be a numpy array with the contents of the input to the
    # tf.function
    for aug in transformations1:
      x=aug(x)
    return x
  @tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
  def tf_aug1(input):
    y = tf.numpy_function(aug1_numpy, [input], tf.float64)
    return y
    
  def aug2_numpy(x): #Jitter 0.5 scale 1
    # x will be a numpy array with the contents of the input to the
    # tf.function
    for aug in transformations2:
      x=aug(x)
    return x
  @tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
  def tf_aug2(input):
    y = tf.numpy_function(aug2_numpy, [input], tf.float64)
    return y
  
  AUTO = tf.data.AUTOTUNE
  SEED = 34
  ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
  #ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
  ssl_ds_one = (
      ssl_ds_one.shuffle(1024, seed=SEED)
      .map(tf_aug1, num_parallel_calls=AUTO)
      .batch(BATCH_SIZE)
      .prefetch(AUTO)
  )
  
  ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train)
  ssl_ds_two = (
      ssl_ds_two.shuffle(1024, seed=SEED)
      .map(tf_aug2, num_parallel_calls=AUTO)
      .batch(BATCH_SIZE)
      .prefetch(AUTO)
  )
  
  # We then zip both of these datasets.
  ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))
  
  mlp_s=2048
  num_training_samples = len(x_train)
  steps = EPOCHS * (num_training_samples // BATCH_SIZE)
  lr_decayed_fn = tf.keras.experimental.CosineDecay(
      initial_learning_rate=5e-5, decay_steps=steps
  )
  
  # Create an early stopping callback.
  early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor="loss", patience=5, restore_best_weights=True, min_delta=0.0001
  )
  
  # Compile model and start training.
  #SGD(lr_decayed_fn, momentum=0.9)
   
  en = get_encoder(frame_size,x_train.shape[-1],mlp_s, config_num=config_num, reg_con=reg_con, origin=origin)
  en.summary()
  
  contrastive = Contrastive(get_encoder(frame_size, x_train.shape[-1], mlp_s, config_num=config_num, reg_con=reg_con, origin=origin), get_predictor(mlp_s, pred_config_num=pred_config_num, origin=origin))
  contrastive.compile(optimizer=tf.keras.optimizers.Adam(lr_decayed_fn))
  
  history = contrastive.fit(ssl_ds, epochs=EPOCHS, callbacks=[early_stopping])
  
  backbone = tf.keras.Model(
      contrastive.encoder.input, contrastive.encoder.output
  )
  
  backbone.summary()
  
  backbone = backbone.layers[1]
  
  backbone.summary()
  
  return backbone