import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
from simsiam import *
from simsiam_multi import *
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import schedules

from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Concatenate, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Conv1DTranspose
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering, FeatureAgglomeration
from sklearn.preprocessing import StandardScaler

from scipy.ndimage import gaussian_filter1d

print(np.__version__)

mlp_s=2048
isall=True
pipe=True
norma=True
filt=False
filt_sigma=10
mix = True
origin=False
graphs=True
sup=False
num_iter=10
x_points=5

window_size=128
num_fet=9

ftr=num_fet
frame_size=window_size
BATCH_SIZE = 128
BATCH_SIZE_1 = 128
AUTO = tf.data.AUTOTUNE

adlr=0.003
EPOCHS_L = 100
EPOCHS = 50

if isall:
    samp_list=["Placeholder"]
else:
    samp_list=[(i+1)*(100//x_points) for i in range(x_points)]
#samp_list=[500]

x_train, y_train, x_test, y_test = data_fetch(norma, filt, filt_sigma, mix)


def custom_augment(data_all):
    out = np.array([])
    for i in range(data_all.shape[0]):
        data = np.copy(data_all[i])
        for i in range(1):
            #augment = [DA_Scaling, DA_Jitter, DA_MagWarp, DA_TimeWarp, DA_Permutation, DA_Combined, DA_Flip, DA_Rotation, DA_Drop]
            augment = [DA_Flip]
            for aug in augment:
                out_temp = aug(data)
                out_temp = np.reshape(out_temp,(1,out_temp.shape[0],out_temp.shape[1]))
                if out.shape[0]==0:
                    out = np.copy(out_temp)
                else:
                    out = np.concatenate((out,out_temp),axis=0)
    return out

def aug1_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Flip(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_aug1(input):
  y = tf.numpy_function(aug1_numpy, [input], tf.float64)
  return y
  
def aug2_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Scaling(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_aug2(input):
  y = tf.numpy_function(aug2_numpy, [input], tf.float64)
  return y
  
def aug3_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Scaling(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_aug3(input):
  y = tf.numpy_function(aug3_numpy, [input], tf.float64)
  return y
  
def aug4_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Scaling(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_aug4(input):
  y = tf.numpy_function(aug4_numpy, [input], tf.float64)
  return y
  
def aug5_numpy(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return DA_Scaling(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def tf_aug5(input):
  y = tf.numpy_function(aug5_numpy, [input], tf.float64)
  return y

if not pipe and not sup:
    print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
    print('x_train',x_train.shape)
    #d1 = scaling(x_train)
    #print('d1',d1.shape)
    #d2= custom_augment(x_train)
    #print('d2',d2.shape)
    print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
    
    
    SEED = 34
    ssl_ds_one = tf.data.Dataset.from_tensor_slices(scaling(x_train))
    #ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_one = (
        ssl_ds_one.shuffle(1024, seed=SEED)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    ssl_ds_two = tf.data.Dataset.from_tensor_slices(custom_augment(x_train))
    ssl_ds_two = (
        ssl_ds_two.shuffle(1024, seed=SEED)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))
elif pipe and not sup:
    
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
    ssl_ds_three = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_three = (
        ssl_ds_three.shuffle(1024, seed=SEED)
        .map(tf_aug3, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    ssl_ds_four = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_four = (
        ssl_ds_four.shuffle(1024, seed=SEED)
        .map(tf_aug4, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    ssl_ds_five = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_five = (
        ssl_ds_five.shuffle(1024, seed=SEED)
        .map(tf_aug5, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    ssl_ds_six = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_six = (
        ssl_ds_six.shuffle(1024, seed=SEED)
        .map(tf_aug5, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    ssl_ds_seven = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_seven = (
        ssl_ds_seven.shuffle(1024, seed=SEED)
        .map(tf_aug5, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    ssl_ds_eight = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_eight = (
        ssl_ds_eight.shuffle(1024, seed=SEED)
        .map(tf_aug5, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    ssl_ds_nine = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_nine = (
        ssl_ds_nine.shuffle(1024, seed=SEED)
        .map(tf_aug5, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    ssl_ds_ten = tf.data.Dataset.from_tensor_slices(x_train)
    ssl_ds_ten = (
        ssl_ds_ten.shuffle(1024, seed=SEED)
        .map(tf_aug5, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two, ssl_ds_three, ssl_ds_four, ssl_ds_five, ssl_ds_six, ssl_ds_seven, ssl_ds_eight, ssl_ds_nine, ssl_ds_ten))
    #ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))
##########################################################################

kappa_arr=np.array([])
acc_arr=np.array([])
max_kappa=0.9460016
for samples_per_class in samp_list:
    con_loss=[]
    kappa=[]
    test_ac=[]
    for count in range(num_iter):  
        # Create a cosine decay learning scheduler.
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
        
        en = get_encoder(frame_size,ftr,mlp_s,origin)
        en.summary()
        
        contrastive = Contrastive_multi(get_encoder(frame_size,ftr,mlp_s,origin), get_predictor(mlp_s,origin))
        contrastive.compile(optimizer=tf.keras.optimizers.Adam(lr_decayed_fn))
        if not sup:
            history = contrastive.fit(ssl_ds, epochs=EPOCHS, callbacks=[early_stopping])
            con_loss.append(history.history['loss'][-1])
        
        
        backbone = tf.keras.Model(
            contrastive.encoder.input, contrastive.encoder.output
        )
        
        backbone.trainable = False
        
        if graphs and count==num_iter-1 and not sup and samp_list.index(samples_per_class)==len(samp_list)-1:
            enc_results = backbone(x_train)
            enc_results = np.array(enc_results)
            #enc_results = np.reshape(enc_results,(enc_results.shape[1], enc_results.shape[0]))
            fig1 = plt.figure(figsize=(25,12))
            plt.boxplot(enc_results)
            plt.savefig('graphs/boxplot1.png')
            #plt.show()
            plt.close(fig1)
            
            # Clustering and visualization of the 
            X_embedded = TSNE(n_components=2).fit_transform(enc_results)
            fig2 = plt.figure(figsize=(18,12))
            plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_train)
            plt.legend()
            plt.savefig('graphs/latentspace1.png')
            plt.close(fig2)
            #plt.show()
        
        BATCH_SIZE = BATCH_SIZE_1
        x_L, y_L, x_L_val, y_L_val  = clas_data_load(samples_per_class, x_train_L=x_train, y_train_L=y_train, isall=isall)

        train_ds = tf.data.Dataset.from_tensor_slices((x_L, y_L))
        val = tf.data.Dataset.from_tensor_slices((x_L_val, y_L_val))
        
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
        backbone.trainable = sup
        inputs = layers.Input((frame_size,ftr))
        x = backbone(inputs, training=sup)
        x = Dense(256, activation='relu')(x)
        act_= Dense(6, activation='softmax', name='act_')(x)
        
        model_C_ = tf.keras.models.Model(inputs, act_, name='classifier')
        
        steps = EPOCHS_L * (num_training_samples // BATCH_SIZE)
        lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=adlr, decay_steps=steps)
        
        opt = keras.optimizers.Adam(learning_rate=adlr)
        model_C_.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
        
        model_C_.summary()
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01,patience=20,restore_best_weights=True )
        
        #history_c_=model_C_.fit(x_L,y_L,epochs=100, validation_data=(x_L_val,y_L_val), shuffle=True, callbacks = callback, batch_size=128)
        #history_c_=model_C_.fit(train_ds,epochs=100, validation_data=val, shuffle=True, callbacks = callback)
        history_c_=model_C_.fit(x_L, y_L, epochs=EPOCHS_L, validation_data=(x_L_val, y_L_val), shuffle=True, callbacks = callback, batch_size=BATCH_SIZE)
        
        plt.figure(1)
        plt.plot(history_c_.history['accuracy'])
        plt.plot(history_c_.history['val_accuracy'])
        plt.title('model accuracy '+str(samples_per_class)+' samples per class')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("graphs/accuracvy plot_classification.png")
        
        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data")
        results = model_C_.evaluate(x_test, y_test)
        #results = model_C_.evaluate(test_ds)
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
        
        if max_kappa<result.numpy():
            max_kappa=result.numpy()
            model_C_.save("models/contrastive_multi")
        
        
        if graphs and count==num_iter-1 and sup and samp_list.index(samples_per_class)==len(samp_list)-1:
            enc_results = backbone(x_test)
            enc_results = np.array(enc_results)
            #enc_results = np.reshape(enc_results,(enc_results.shape[1], enc_results.shape[0]))
            fig3 = plt.figure(figsize=(25,12))
            plt.boxplot(enc_results)
            plt.savefig('graphs/boxplot2.png')
            #plt.show()
            plt.close(fig3)
            
            # Clustering and visualization of the 
            X_embedded = TSNE(n_components=2).fit_transform(enc_results)
            fig4 = plt.figure(figsize=(18,12))
            plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_test, label=y_test)
            plt.legend()
            plt.savefig('graphs/latentspace2.png')
            plt.close(fig4)
            #plt.show()
    
    print('*****************************************************************************************************************************************************************************************************************')
    
    print('test_acc')
    for el in test_ac:
      print(el)
    print('kappa')
    for el in kappa:
      print(el)
    print('final con loss')
    for el in con_loss:
      print(el)
    
    if kappa_arr.shape[0]==0:
        kappa_arr = np.array([kappa])
        acc_arr = np.array([test_ac])
    else:
        kappa_arr = np.append(kappa_arr,[kappa],axis=0)
        acc_arr = np.append(acc_arr,[test_ac],axis=0)
        
print(kappa_arr)
print(acc_arr)
print('max_kappa: ',max_kappa)
#np.savez('data/graph_data_cont.npz', kappa_arr, acc_arr)