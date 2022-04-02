import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
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
ext=False
train_trunk=False
mthd='train'
x_train=np.array([])

model=keras.models.load_model("best_model_0.8516")

labled=np.load('best_data.npz')
x_L=labled['arr_0']
y_L=labled['arr_1']
x_L_val=labled['arr_2']
y_L_val=labled['arr_3']

from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Dropout

num_fet=3

input_c_ = Input(shape=(window_size,num_fet), name='input_c_')

fet_extrct=model.layers[7]

fet_extrct.trainable=train_trunk

fet_=fet_extrct(input_c_)

pool_C_ = MaxPool1D(2,name='pool_C_')(fet_)
flat_C_ = Flatten(name="flat_C_")(pool_C_)

dens_C_1 = Dense(1024, activation='relu', name='dens_C_1')(flat_C_)
act_= Dense(6, activation='softmax', name='act_')(dens_C_1)

model_C_ = tf.keras.models.Model(input_c_, act_, name='classifier')

opt = keras.optimizers.Adam(learning_rate=0.0003)
model_C_.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

model_C_.summary()

#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.05,patience=15,restore_best_weights=True )

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

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model_C_.evaluate(x_test, y_test)
print("test acc:", results[1]*100,"%")

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

model_C_.save("full model")