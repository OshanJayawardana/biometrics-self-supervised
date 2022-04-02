import pandas as pd
from math import gcd
import numpy as np
import matplotlib.pyplot as plt
from transformations import *
import tensorflow as tf
import tensorflow_addons as tfa
import keras
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers
from tensorflow.keras.optimizers import schedules

from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Dropout

print(np.__version__)

kappa=[]
test_ac=[]
print('*****************************************************************************************************************************************************************************************************************')
for i in range(1):
    window_size=128
    num_fet=3
    samples_per_class=100
    
    ftr=num_fet
    frame_size=window_size
    BATCH_SIZE = 512
    
    mthd='train'
    x_train=np.array([])
    for atrbt in ['acc']:
        for axs in ['x','y','z']:
            #path=str(os.path.abspath(os.getcwd()))
            #path+=r"\UCI HAR\uci-human-activity-recognition\original\UCI HAR Dataset\train\Inertial Signals"
            path=r"UCI HAR/uci-human-activity-recognition/original/UCI HAR Dataset/train/Inertial Signals"
            file = open(path+r"/total_"+atrbt+"_"+axs+"_"+mthd+".txt",'r')
            data=(np.array(file.read().split()))
            data=data.reshape(len(data)//window_size,window_size)
            data=data.reshape(data.shape[0],data.shape[1],1)
            data=data.astype(float)
            #data=data[:,:data.shape[1]//2,:]
            if x_train.size==0:
                x_train=data
            else:
                x_train=np.append(x_train,data,axis=2)
    
    file=open(r"UCI HAR/uci-human-activity-recognition/original/UCI HAR Dataset/train/Inertial Signals/y_train.txt",'r')
    y_train=np.array(file.read().split())
    y_train=y_train.astype(int)
    y_train=y_train-1
    
    num_sample=x_train.shape[0]
    
    print(x_train.shape)
    print(y_train.shape)
    
    def custom_augment0(data):
        #np.random.seed(rand)
        # As discussed in the SimCLR paper, the series of augmentation
        # transformations (except for random crops) need to be applied
        # randomly to impose translational invariance.
        
        #data = tf_Flip(data)
        #data = DA_Jitter(data,5)
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
        
        #x = layers.Flatten()(inputs)
        #x = layers.BatchNormalization()(x)
        #x = layers.Dense(frame_size*ftr, kernel_regularizer=regularizers.l2(0.0001))(x)
        #x = layers.Reshape((frame_size,ftr), input_shape=(frame_size*ftr,))(x)
        
        #x = layers.Conv1D(filters=32, kernel_size=24,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(x)#64 #10
        #x = layers.Dropout(0.1)(x)
        #x = layers.Conv1D(filters=64, kernel_size=16,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(x)#128 #5
        #x = layers.Dropout(0.1)(x)
        x = layers.Conv1D(filters=96, kernel_size=3,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(inputs)#128 #5
        x = layers.Dropout(0.1)(x)
        
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Flatten(name='flat')(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.Dense(mlp_s, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(mlp_s, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(mlp_s, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
        
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
    
    # Visualize the training progress of the model.
    #f1=plt.figure()
    #plt.plot(history.history["loss"])
    #plt.grid()
    #plt.title("Negative Cosine Similairty")
    #plt.show()
    window_size=128
    mthd='train'
    labled=np.load('best_data.npz')
    x_L=labled['arr_0']
    y_L=labled['arr_1']
    x_L_val=labled['arr_2']
    y_L_val=labled['arr_3']
    
    # Then we shuffle, batch, and prefetch this dataset for performance. We
    # also apply random resized crops as an augmentation but only to the
    # training set.
    # Extract the backbone
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_L, y_L))
    val = tf.data.Dataset.from_tensor_slices((x_L_val, y_L_val))
    
    # Then we shuffle, batch, and prefetch this dataset for performance. We
    # also apply random resized crops as an augmentation but only to the
    # training set.
    train_ds = (
        train_ds.shuffle(1024)
        .map(lambda x, y: (x, y), num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    val = val.batch(BATCH_SIZE).prefetch(AUTO)
    
    backbone = tf.keras.Model(
        contrastive.encoder.input, contrastive.encoder.output
    )
    
    # We then create our linear classifier and train it.
    backbone.trainable = False
    inputs = layers.Input((frame_size,ftr))
    x = backbone(inputs, training=False)
    x = Dense(1024, activation='relu')(x)
    act_= Dense(6, activation='softmax', name='act_')(x)
    
    model_C_ = tf.keras.models.Model(inputs, act_, name='classifier')
    
    opt = keras.optimizers.Adam(learning_rate=0.0003)
    model_C_.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    
    model_C_.summary()
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.05,patience=15,restore_best_weights=True )
    
    history_c_=model_C_.fit(train_ds,epochs=100, validation_data=val, shuffle=True)
    
    plt.figure(1)
    plt.plot(history_c_.history['accuracy'])
    plt.plot(history_c_.history['val_accuracy'])
    plt.title('model accuracy '+str(samples_per_class)+' samples per class')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("accuracvy plot_classification.png")
    
    mthd='test'
    x_test=np.array([])
    for atrbt in ['acc']:
        for axs in ['x','y','z']:
            path=r"UCI HAR/uci-human-activity-recognition/original/UCI HAR Dataset/test/Inertial Signals"
            file = open(path+r"/total_"+atrbt+"_"+axs+"_"+mthd+".txt",'r')
            data=np.array(file.read().split())
            data=data.reshape(len(data)//window_size,window_size)
            data=data.reshape(data.shape[0],data.shape[1],1)
            data=data.astype(float)
            if x_test.size==0:
                x_test=data
            else:
                x_test=np.append(x_test,data,axis=2)
                
    #x_test=stats.zscore(x_trest, axis=2, ddof=0)
    
    file=open(r"UCI HAR/uci-human-activity-recognition/original/UCI HAR Dataset/test/Inertial Signals/y_test.txt",'r')
    y_test=np.array(file.read().split())
    y_test=y_test.astype(int)
    y_test=y_test-1
    
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTO)
    
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model_C_.evaluate(test_ds)
    print("test acc:", results[1]*100,"%")
    test_ac.append(results[1]*100)
    
    off=10
    start=895
    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for "+str(off)+" samples")
    predictions = model_C_.predict(x_test[start:start+off])
    print("True labels: ",y_test[start:start+off])
    y_pred_test=[]
    for i in range(off):
        y_pred_test.append((np.where(predictions[i] == np.amax(predictions[i])))[0][0])
    print("pred labels: ",np.array(y_pred_test))
    
    #Calculating kappa score
    metric = tfa.metrics.CohenKappa(num_classes=6, sparse_labels=True)
    metric.update_state(y_true=y_test , y_pred=model_C_.predict(x_test))
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