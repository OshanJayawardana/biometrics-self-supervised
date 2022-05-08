import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE
from projectors import *
from predictors import *
from simsiam import *

from backbones import *
from data_loader import *

def pre_trainer(scen, fet):
  BATCH_SIZE = 40
  origin = False
  EPOCHS = 100
  frame_size   = 128
  path = "/home/oshanjayawardanav100/biometrics-self-supervised/gait_dataset/idnet/"
  
  users_2 = list(range(17,51)) #Users for dataset 2
  users_1 = list(range(1,17)) #Users for dataset 1
  
  x_train, y_train, x_val, y_val, x_test, y_test, sessions = data_loader_gait(path, classes=users_2, frame_size=frame_size)
  
  x_train = norma_pre(x_train)
  print("x_train", x_train.shape)
  
  def aug1_numpy(x):
    # x will be a numpy array with the contents of the input to the
    # tf.function
    x = DA_Jitter(x, 0.5)
    return DA_Scaling(x, 1)
  @tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
  def tf_aug1(input):
    y = tf.numpy_function(aug1_numpy, [input], tf.float64)
    return y
    
  def aug2_numpy(x):
    # x will be a numpy array with the contents of the input to the
    # tf.function
    x = DA_Jitter(x, 0.5)
    return DA_Scaling(x, 1)
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
  
  en = get_encoder(frame_size,x_train.shape[-1],mlp_s,origin)
  en.summary()
  
  contrastive = Contrastive(get_encoder(frame_size,x_train.shape[-1],mlp_s,origin), get_predictor(mlp_s,origin))
  contrastive.compile(optimizer=tf.keras.optimizers.Adam(lr_decayed_fn))
  
  history = contrastive.fit(ssl_ds, epochs=EPOCHS, callbacks=[early_stopping])
  
  backbone = tf.keras.Model(
      contrastive.encoder.input, contrastive.encoder.output
  )
  
  backbone.summary()
  
  backbone = tf.keras.Model(backbone.input, backbone.layers[-fet].output)
  
  backbone.summary()
  
  x_train, y_train, x_val, y_val, x_test, y_test, sessions = data_loader_gait(path, classes=users_2, frame_size=frame_size)
    
  x_train = norma_pre(x_train)
  enc_results = backbone(x_train)
  enc_results = np.array(enc_results)
  X_embedded = TSNE(n_components=2).fit_transform(enc_results)
  fig4 = plt.figure(figsize=(18,12))
  plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_train)
  plt.savefig('graphs/latentspace_scen_1.png')
  plt.close(fig4)
  
  return backbone