import pandas as pd
from math import gcd
import numpy as np
import matplotlib.pyplot as plt
from transformations import *
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import schedules

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Dropout

from data_loader import *

print(np.__version__)

kappa=[]
test_ac=[]
print('*****************************************************************************************************************************************************************************************************************')
for i in range(10):
    ftr=24
    frame_size=30
    BATCH_SIZE = 40
    
    path = "/home/oshanjayawardanav100/biometrics-self-supervised/musicid_dataset/"
    users_2 = list(range(9,21)) #Users for dataset 2
    users_1 = users = list(range(1,7)) #Users for dataset 1
    folder_train = ["TrainingSet","TestingSet_secret", "TestingSet"]
  
    x_train, y_train, sessions_train = data_load_origin(path, users=users_1, folders=folder_train, frame_size=30)
    
    num_sample=x_train.shape[0]
    
    print(x_train.shape)
    print(y_train.shape)
    
    def custom_augment0(data):
        #np.random.seed(rand)
        # As discussed in the SimCLR paper, the series of augmentation
        # transformations (except for random crops) need to be applied
        # randomly to impose translational invariance.
        
        #data = tf_Flip(data)
        data = DA_Jitter(data, 0.5)
        data = DA_Scaling(data, 1)
        
        #ret_data[i] = DA_MagWarp(data, 0.1)
        #data = DA_TimeWarp(data, 0.5)
        #ret_data[i] = DA_Permutation(data,minSegLength=1)
        return data
    
    def custom_augment(data_all):
        #np.random.seed(rand)
        # As discussed in the SimCLR paper, the series of augmentation
        # transformations (except for random crops) need to be applied
        # randomly to impose translational invariance.
        ret_data=np.copy(data_all)
        for i in range(ret_data.shape[0]):
            data = ret_data[i]
            data = DA_Scaling(data, 1)
            ret_data[i] = DA_Jitter(data,0.05)
            #ret_data[i] = DA_MagWarp(data, 0.1)
            #data = DA_TimeWarp(data, 0.5)
            #ret_data[i] = DA_Permutation(data,minSegLength=1)
        return ret_data
    
    AUTO = tf.data.AUTOTUNE
    SEED = 34
    ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
    #ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_one = (
        ssl_ds_one.shuffle(1024, seed=SEED)
        .map(custom_augment0, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_two = (
        ssl_ds_two.shuffle(1024, seed=SEED)
        .map(custom_augment0, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    # We then zip both of these datasets.
    ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))
    
    mlp_s=2048
    
    def get_encoder():
        # Input and backbone.
        inputs = layers.Input((frame_size,ftr))
        
        #x = keras.backend.permute_dimensions(inputs, pattern=(0,2,1))
        #x = keras.layers.Lambda(lambda v: tf.math.real(tf.signal.rfft(v)))(x)
        #x = keras.backend.permute_dimensions(x, pattern=(0,2,1))
        #x = layers.Conv1D(filters=16, kernel_size=5, activation='relu',kernel_regularizer=regularizers.l2(0.0001))(inputs)#64 #10
        #x = layers.Dropout(0.1)(x)
        #x = layers.Conv1D(filters=32, kernel_size=5, activation='relu',kernel_regularizer=regularizers.l2(0.0001))(inputs)#64 #10
        #x = layers.Dropout(0.1)(x)
        x = layers.Conv1D(filters=64, kernel_size=5, activation='relu',kernel_regularizer=regularizers.l2(0.0001))(inputs)#64 #10
        x = layers.Dropout(0.1)(x)
        x = layers.Flatten(name='flat')(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.Dense(mlp_s, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(mlp_s, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(mlp_s, kernel_regularizer=regularizers.l2(0.0001))(x)
        
        
        return tf.keras.Model(inputs, outputs, name="encoder")
        
    en=get_encoder()
    en.summary()
    
    def get_predictor():
        model = tf.keras.Sequential(
            [   
                layers.Input((mlp_s,)),
                layers.BatchNormalization(),
                layers.Dense(
                    mlp_s//4,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(1e-4),
                ),
                
                layers.Dense(
                    mlp_s,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(1e-4),
                ),
            ],
            name="predictor",
        )
        return model
        
    pr=get_predictor()
    pr.summary()
    
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

    class Contrastive(tf.keras.Model):
        def __init__(self, encoder, predictor):
            super(Contrastive, self).__init__()
            self.encoder = encoder
            self.predictor = predictor
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
        @property
        def metrics(self):
            return [self.loss_tracker]
    
        def train_step(self, data):
            # Unpack the data.
            ds_one, ds_two = data
    
            # Forward pass through the encoder and predictor.
            with tf.GradientTape() as tape:
                z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
                p1, p2 = self.predictor(z1), self.predictor(z2)
                # Note that here we are enforcing the network to match
                # the representations of two differently augmented batches
                # of data.
                loss = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2
    
            # Compute gradients and update the parameters.
            learnable_params = (
                self.encoder.trainable_variables + self.predictor.trainable_variables
            )
            gradients = tape.gradient(loss, learnable_params)
            self.optimizer.apply_gradients(zip(gradients, learnable_params))
    
            # Monitor loss.
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}
    
    EPOCHS = 100
    # Create a cosine decay learning scheduler.
    num_training_samples = len(x_train)
    print("num_training_samples",num_training_samples)
    steps = EPOCHS * (num_training_samples // BATCH_SIZE)
    print("steps",steps)
    lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=0.05, decay_steps=steps
    )
    
    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True
    )
    
    # Compile model and start training.
    contrastive = Contrastive(get_encoder(), get_predictor())
    contrastive.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9))
    #history = contrastive.fit(ssl_ds, epochs=EPOCHS, callbacks=[early_stopping])
    history = contrastive.fit(ssl_ds, epochs=EPOCHS)
    
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
    
    # Visualize the training progress of the model.
    #f1=plt.figure()
    #plt.plot(history.history["loss"])
    #plt.grid()
    #plt.title("Negative Cosine Similairty")
    #plt.show()
    
    # Then we shuffle, batch, and prefetch this dataset for performance. We
    # also apply random resized crops as an augmentation but only to the
    # training set.
    
    # Extract the backbone
    backbone = tf.keras.Model(
        contrastive.encoder.input, contrastive.encoder.output
    )
    
    # We then create our linear classifier and train it.
    backbone.trainable = False
    inputs = layers.Input((frame_size,ftr))
    x = backbone(inputs, training=False)
    #x = layers.Dense(8192, activation="relu")(x)
    #x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    linear_model = tf.keras.Model(inputs, outputs, name="linear_model")
    
    # Compile model and start training.
    linear_model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9),
    )
    
    linear_model.summary()
    
    #history = linear_model.fit(train_ds, epochs=EPOCHS)
    history = linear_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS)
        
    
    #_, test_acc = linear_model.evaluate(test_ds)
    _, test_acc = linear_model.evaluate(x_test, y_test)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))
    test_ac.append(test_acc * 100)
    
    x_test_ds = tf.data.Dataset.from_tensor_slices(x_test)
    x_test_ds = x_test_ds.batch(BATCH_SIZE).prefetch(AUTO)
    
    #y_pred = linear_model.predict(x_test_ds)
    y_pred = linear_model.predict(x_test)
    y_pred_c = []
    
    for i in range(y_pred.shape[0]):
        y_pred_c.append(np.argmax(y_pred[i]))
        
    y_pred_c = np.array(y_pred_c)
    
    #print(y_test.shape)
    #print(y_test)
    #print(y_pred_c.shape)
    #print(y_pred_c)
    
    #Calculating kappa score
    metric = tfa.metrics.CohenKappa(num_classes=15, sparse_labels=True)
    metric.update_state(y_true=y_test , y_pred=y_pred_c)
    result = metric.result()
    print('kappa score: ',result.numpy())
    kappa.append(result.numpy())
    
    #f2=plt.figure()
    #plt.plot(history.history["loss"])
    #plt.grid()
    #plt.title("Classification loss")
    #plt.show()
print('*****************************************************************************************************************************************************************************************************************')
for el in test_ac:
  print(el)
  
for el in kappa:
  print(el)