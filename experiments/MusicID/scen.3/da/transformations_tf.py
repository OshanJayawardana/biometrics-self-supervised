from transformations import *

def flip_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Flip(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_flip(input):
  y = tf.numpy_function(flip_numpy, [input], tf.float64)
  return y
  
def scale_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Scaling(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_scale(input):
  y = tf.numpy_function(scale_numpy, [input], tf.float64)
  return y
  
def jitter_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Jitter(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_jitter(input):
  y = tf.numpy_function(jitter_numpy, [input], tf.float64)
  return y
  
def magwarp_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_MagWarp(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_magwarp(input):
  y = tf.numpy_function(magwarp_numpy, [input], tf.float64)
  return y
  
def timewarp_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_TimeWarp(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_timewarp(input):
  y = tf.numpy_function(timewarp_numpy, [input], tf.float64)
  return y
  
def permutation_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Permutation(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_permutation(input):
  y = tf.numpy_function(permutation_numpy, [input], tf.float64)
  return y