import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Concatenate, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout

from backbones import *
from data_load import *

isall=True
domain = 0

if domain:
    num_classes=6
else:
    num_classes=30

if isall:
    num_samp = ["Placeholder"]
else:
    num_samp = [5,10,15,20,25,35,45,55,65,75,95,115,135,155,175,215,255,295,335,375,455,535,615,695,775,935,1000]

x_train, y_train, x_test, y_test = data_fetch(norma=True, filt=False, filt_sigma=5, mix=True, domain=domain)

kappa_arr=np.array([])
acc_arr=np.array([])
for samples_per_class in num_samp:
    kappa=[]
    test_ac=[]
    for count in range(1):
        x_L, y_L, x_L_val, y_L_val  = clas_data_load(samples_per_class, x_train_L=x_train, y_train_L=y_train, isall=isall, domain=domain)
        ks = 3
        con =4
        inputs = Input(shape=(128,9))
        x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling1D(pool_size=4, strides=4)(x)
        x = Dropout(rate=0.1)(x)
        x = resnetblock(x, CR=32*con, KS=ks)
        x = resnetblock(x, CR=32*con, KS=ks)
        x = resnetblock(x, CR=64*con, KS=ks)
        x = resnetblock(x, CR=64*con, KS=ks)
        x = resnetblock(x, CR=128*con, KS=ks)
        x = resnetblock(x, CR=128*con, KS=ks)
        x = resnetblock_final(x, CR=128*con, KS=ks)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        resnettssd = Model(inputs, outputs)
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=5)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.0001, decay_rate=0.5, decay_steps=1000000)# 0.0001, 0.9, 100000
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        resnettssd.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
        history = resnettssd.fit(x_L, y_L, validation_data=(x_L_val, y_L_val), epochs=100, callbacks=callback, batch_size=128)
        
        results = resnettssd.evaluate(x_test,y_test)
        print("test acc:", results[1])
        
        #Calculating kappa score
        metric = tfa.metrics.CohenKappa(num_classes=num_classes, sparse_labels=True)
        metric.update_state(y_true=y_test , y_pred=resnettssd.predict(x_test))
        result = metric.result()
        print('kappa score: ',result.numpy())
        
        test_ac.append(results[1]*100)
        kappa.append(result.numpy())
    if kappa_arr.shape[0]==0:
        kappa_arr = np.array([kappa])
        acc_arr = np.array([test_ac])
    else:
        kappa_arr = np.append(kappa_arr,[kappa],axis=0)
        acc_arr = np.append(acc_arr,[test_ac],axis=0)
print(kappa_arr)
print(acc_arr)
np.savez('data/graph_data_resnet.npz', kappa_arr, acc_arr)
