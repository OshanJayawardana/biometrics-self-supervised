import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from scipy import stats
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
from transformations import *
import random

from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Dropout

window_size=128
num_fet=3
samples_per_class=100
ext=True


transformations=np.array([DA_Jitter, DA_Scaling, DA_MagWarp, DA_TimeWarp, DA_Rotation, DA_Permutation, DA_Combined])
samp_list = []
for i in range(20):
    samp_list.append(5*(i+1))

kappa_arr=np.array([])
acc_arr=np.array([])
for samples_per_class in samp_list:
    kappa=[]
    test_acc=[]
    for i in range(10):                
        #x_train_pro=stats.zscore(x_train_pro, axis=3, ddof=0)
        
        def trunk():
            input_ = Input(shape=(window_size,num_fet), name='input_')
            conv_1 = Conv1D(32, 24, activation='relu', name='conv_1', kernel_regularizer=regularizers.l2(0.0001))(input_)
            drop_1 = Dropout(0.1, name='drop_1')(conv_1)
        
            conv_2 = Conv1D(64, 16, activation='relu', name='conv_2', kernel_regularizer=regularizers.l2(0.0001))(drop_1)
            drop_2 = Dropout(0.1, name='drop_2')(conv_2)
        
            conv_3 = Conv1D(96, 8, activation='relu', name='conv_3', kernel_regularizer=regularizers.l2(0.0001))(drop_2)
            drop_3 = Dropout(0.1, name='drop_3')(conv_3)
        
            return tf.keras.models.Model(input_,drop_3,name='trunk_')
        
        
        trunk=trunk()
        
        window_size=128
        mthd='train'
        x_train_L=np.array([])
        for atrbt in ['acc']:
            for axs in ['x','y','z']:
                path=r"UCI HAR/uci-human-activity-recognition/original/UCI HAR Dataset/train/Inertial Signals"
                file = open(path+r"/body_"+atrbt+"_"+axs+"_"+mthd+".txt",'r')
                data=np.array(file.read().split())
                data=data.reshape(len(data)//window_size,window_size)
                data=data.reshape(data.shape[0],data.shape[1],1)
                data=data.astype(float)
                if x_train_L.size==0:
                    x_train_L=data
                else:
                    x_train_L=np.append(x_train_L,data,axis=2)
        #x_train_L=stats.zscore(x_train_L, axis=2, ddof=0)
        
        file=open(r"UCI HAR/uci-human-activity-recognition/original/UCI HAR Dataset/train/Inertial Signals/y_train.txt",'r')
        y_train_L=np.array(file.read().split())
        y_train_L=y_train_L.astype(int)
        y_train_L=y_train_L-1
        
        indx=np.arange(x_train_L.shape[0])
        np.random.shuffle(indx)
        
        x_train_L=x_train_L[indx]
        y_train_L=y_train_L[indx]
        
        cnt=[0,0,0,0,0,0]
        x_L=np.array([])
        y_L=np.array([])
        for i in range(x_train_L.shape[0]):
            for j in range(6):
                if y_train_L[i]==j and cnt[j]<samples_per_class:
                    if x_L.shape[0]==0:
                        x_L=x_train_L[i].reshape((1,128,3))
                    else:
                        x_L=np.append(x_L,x_train_L[i].reshape((1,128,3)),axis=0)
                    y_L=np.append(y_L,j)
                    cnt[j]+=1
            if sum(cnt)>=samples_per_class*6:
                break
        
        print(i)
        
        cnt=[0,0,0,0,0,0]
        x_L_val=np.array([])
        y_L_val=np.array([])
        for k in range(i,x_train_L.shape[0]):
            for j in range(6):
                if y_train_L[k]==j and cnt[j]<(samples_per_class*20/80):
                    if x_L_val.shape[0]==0:
                        x_L_val=x_train_L[k].reshape((1,128,3))
                    else:
                        x_L_val=np.append(x_L_val,x_train_L[k].reshape((1,128,3)),axis=0)
                    y_L_val=np.append(y_L_val,j)
                    cnt[j]+=1
            if sum(cnt)>=(samples_per_class*20/80)*6:
                break
        
        from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Dropout
        
        num_fet=3
        
        input_C_ = Input(shape=(window_size,num_fet), name='input_C_')
        
        fet_=trunk(input_C_)
        
        pool_C_ = MaxPool1D(2,name='pool_C_')(fet_)
        flat_C_ = Flatten(name="flat_C_")(pool_C_)
        
        dens_C_1 = Dense(1024, activation='relu', name='dens_C_1')(flat_C_)
        act_= Dense(6, activation='softmax', name='act_')(dens_C_1)
        
        model_C_ = tf.keras.models.Model(input_C_, act_, name='classifier')
        
        opt = keras.optimizers.Adam(learning_rate=0.0003)
        model_C_.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
        
        model_C_.summary()
        
        history_c_=model_C_.fit(x_L,y_L,epochs=100, validation_data=(x_L_val,y_L_val), shuffle=True)
        
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
                file = open(path+r"/body_"+atrbt+"_"+axs+"_"+mthd+".txt",'r')
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
        
        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data")
        results = model_C_.evaluate(x_test, y_test)
        print("test acc:", results[1]*100,"%")
        test_acc.append(results[1]*100)
        
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
    if kappa_arr.shape[0]==0:
        kappa_arr = np.array([kappa])
        acc_arr = np.array([test_acc])
    else:
        kappa_arr = np.append(kappa_arr,[kappa],axis=0)
        acc_arr = np.append(acc_arr,[test_acc],axis=0)
        
print(kappa_arr)
print(acc_arr)
np.savez('graph_data_supervised.npz', kappa_arr, acc_arr)