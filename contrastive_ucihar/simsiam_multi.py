

import pandas as pd
from math import gcd
import numpy as np
import matplotlib.pyplot as plt
from transformations import *
from backbones import *
from projectors import *
from predictors import *
from transformations_Tian import *
from data_load import *
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import schedules

def get_encoder(frame_size,ftr,mlp_s,origin):
    # Input and backbone.
    inputs = layers.Input((frame_size,ftr))
    
    if origin:
        x = CAE_Origin(inputs)
    else:
        x = CAE_multi(inputs)
    x = layers.Flatten(name='flat')(x)
    
    outputs = proTian(x,mlp_s=mlp_s)
    
    return tf.keras.Model(inputs, outputs, name="encoder")
    

def get_predictor(mlp_s, origin):
    
    inputs = layers.Input((mlp_s//4,))
    
    if origin:
        outputs = predTian_Origin(inputs, mlp_s=mlp_s)
    else:
        outputs = predTian(inputs, mlp_s=mlp_s)
    
    return tf.keras.Model(inputs, outputs, name="predictor")
    

def compute_loss(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    #print(p.shape,z.shape)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))
       
class Contrastive_multi(tf.keras.Model):
    def __init__(self, encoder, predictor):
        super(Contrastive_multi, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data.
        ds_1, ds_2, ds_3, ds_4, ds_5, ds_6, ds_7, ds_8, ds_9, ds_10 = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2, z3, z4, z5, z6, z7, z8, z9, z10 = self.encoder(ds_1), self.encoder(ds_2), self.encoder(ds_3), self.encoder(ds_4), self.encoder(ds_5), self.encoder(ds_6), self.encoder(ds_7), self.encoder(ds_8), self.encoder(ds_9), self.encoder(ds_10)
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = self.predictor(z1), self.predictor(z2), self.predictor(z3), self.predictor(z4), self.predictor(z5), self.predictor(z6), self.predictor(z7), self.predictor(z8), self.predictor(z9), self.predictor(z10)
            # Note that here we are enforcing the network to match
            # the representations of two differently augmented batches
            # of data.
            loss = (compute_loss(p1, z2)/18 + compute_loss(p1, z3)/18 + compute_loss(p1, z4)/18 + compute_loss(p1, z5)/18 + compute_loss(p1, z6)/18 + compute_loss(p1, z7)/18 + compute_loss(p1, z8)/18 + compute_loss(p1, z9)/18 + compute_loss(p1, z10)/18
            + compute_loss(p2, z1)/18
            + compute_loss(p3, z1)/18
            + compute_loss(p4, z1)/18
            + compute_loss(p5, z1)/18
            + compute_loss(p6, z1)/18
            + compute_loss(p7, z1)/18
            + compute_loss(p8, z1)/18
            + compute_loss(p9, z1)/18
            + compute_loss(p10, z1)/18)
            
            #loss = (compute_loss(p1, z2)/20 + compute_loss(p1, z3)/20 + compute_loss(p1, z4)/20 + compute_loss(p1, z5)/20 
            #+ compute_loss(p2, z1)/20 + compute_loss(p2, z3)/20 + compute_loss(p2, z4)/20 + compute_loss(p2, z5)/20
            #+ compute_loss(p3, z1)/20 + compute_loss(p3, z2)/20 + compute_loss(p3, z4)/20 + compute_loss(p3, z5)/20
            #+ compute_loss(p4, z1)/20 + compute_loss(p4, z2)/20 + compute_loss(p4, z3)/20 + compute_loss(p4, z5)/20
            #+ compute_loss(p5, z1)/20 + compute_loss(p5, z2)/20 + compute_loss(p5, z3)/20 + compute_loss(p5, z4)/20)

        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables + self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}