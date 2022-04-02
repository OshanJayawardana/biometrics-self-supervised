from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from data_load import *

mlp_s=256
norma=True
mix=True
sup=True
isall=False

window_size=128
num_fet=9

ftr=num_fet
frame_size=window_size
BATCH_SIZE = 128
AUTO = tf.data.AUTOTUNE

adlr=0.003
EPOCHS_L = 100

x_train, y_train, x_test, y_test = data_fetch(norma=norma, filt=False, filt_sigma=5, mix=mix)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

num_samp = [5,10,15,20,25,35,45,55,65,75,95,115,135,155,175,215,255,295,335,375,455,535,615,695,775,935,1000]
print(len(num_samp))
kappa_arr=np.array([])
acc_arr=np.array([])
for samples_per_class in num_samp:
    kappa=[]
    test_ac=[]
    for count in range(10):
        model_temp = keras.models.load_model('./models/contrastive')
        backbone=model_temp.layers[1]

        x_L, y_L, x_L_val, y_L_val = clas_data_load(samples_per_class, x_train, y_train, isall=isall)

        backbone.trainable = sup
        inputs = layers.Input((frame_size,ftr))
        x = backbone(inputs, training=sup)
        x = Dense(256, activation='relu')(x)
        act_= Dense(6, activation='softmax', name='act_')(x)

        model_C_ = tf.keras.models.Model(inputs, act_, name='classifier')

        num_training_samples = x_train.shape[0]
        steps = EPOCHS_L * (num_training_samples // BATCH_SIZE)
        lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=adlr, decay_steps=steps)

        opt = keras.optimizers.Adam(learning_rate=adlr)
        model_C_.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

        #model_C_.summary()

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01,patience=20,restore_best_weights=True )

        #history_c_=model_C_.fit(x_L,y_L,epochs=100, validation_data=(x_L_val,y_L_val), shuffle=True, callbacks = callback, batch_size=128)
        #history_c_=model_C_.fit(train_ds,epochs=100, validation_data=val, shuffle=True, callbacks = callback)
        history_c_=model_C_.fit(x_L, y_L,epochs=EPOCHS_L, validation_data=(x_L_val, y_L_val), shuffle=True, callbacks = callback, batch_size=BATCH_SIZE, verbose=False)

        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data")
        results = model_C_.evaluate(x_test, y_test)
        #results = model_C_.evaluate(test_ds)
        print("test acc:", results[1]*100,"%")
        test_ac.append(results[1]*100)

        #Calculating kappa score
        metric = tfa.metrics.CohenKappa(num_classes=6, sparse_labels=True)
        metric.update_state(y_true=y_test , y_pred=model_C_.predict(x_test))
        result = metric.result()
        print('kappa score: ',result.numpy())
        kappa.append(result.numpy())
    if kappa_arr.shape[0]==0:
        kappa_arr = np.array([kappa])
        acc_arr = np.array([test_ac])
    else:
        kappa_arr = np.append(kappa_arr,[kappa],axis=0)
        acc_arr = np.append(acc_arr,[test_ac],axis=0)

print(kappa_arr)
print(acc_arr)
np.savez('data/graph_data_simsiam_sup.npz', kappa_arr, acc_arr)