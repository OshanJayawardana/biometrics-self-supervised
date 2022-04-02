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

#transformations=np.array([DA_Jitter, DA_Scaling, DA_MagWarp, DA_TimeWarp, DA_Rotation, DA_Permutation, DA_Combined])
#sigma_l=np.array([0.05, 1, 2, 2, None, None, None])

transformations=np.array([DA_Flip])
sigma_l=np.array([None])


indx=list(range(len(transformations)))
random.shuffle(indx)
transformations=transformations[indx]
sigma_l=sigma_l[indx]

if ext:
  m_=8
else:
  m_=2

x_train_pro=np.zeros((len(transformations),num_sample*m_,window_size,3))
y_train_pro=np.zeros((len(transformations),num_sample*m_))

for j in range(num_sample):
    x_train_temp=np.copy(x_train[j])
    for Jt,sigma,i in zip(transformations,sigma_l,range(len(transformations))):
        x_train_pro[i,j*m_,:,:]=np.copy(x_train_temp)
        y_train_pro[i,j*m_]=False
        x_train_pro[i,j*m_+1,:,:]=np.copy(Jt(x_train_temp,sigma=sigma))
        y_train_pro[i,j*m_+1]=True
        if ext:
          cnt=1
          for k in range(len(transformations)):
              if i!=k:
                  x_train_pro[i,j*m_+1+cnt,:,:]=np.copy(transformations[k](x_train_temp,sigma=sigma_l[k]))
                  y_train_pro[i,j*m_+1+cnt]=False
                  cnt+=1
                
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

def global_max_pool():
    input_I = Input(shape=(83,96), name='input_I')
    pool_ = MaxPool1D(2,name='pool_')(input_I)
    flat_ = Flatten(name="flat_")(pool_)
    return tf.keras.models.Model(input_I,flat_,name='global_max_pool_')


input_1 = Input(shape=(window_size,num_fet), name='input_1')
#input_2 = Input(shape=(window_size,num_fet), name='input_2')
#input_3 = Input(shape=(window_size,num_fet), name='input_3')
#input_4 = Input(shape=(window_size,num_fet), name='input_4')
#input_5 = Input(shape=(window_size,num_fet), name='input_5')
#input_6 = Input(shape=(window_size,num_fet), name='input_6')
#input_7 = Input(shape=(window_size,num_fet), name='input_7')
#input_8 = Input(shape=(window_size,num_fet), name='input_8')

trunk=trunk()

fet_1 = trunk(input_1)
#fet_2 = trunk(input_2)
#fet_3 = trunk(input_3)
#fet_4 = trunk(input_4)
#fet_5 = trunk(input_5)
#fet_6 = trunk(input_6)
#fet_7 = trunk(input_7)
#fet_8 = trunk(input_8)

global_max_pool=global_max_pool()

flat_1=global_max_pool(fet_1)
#flat_2=global_max_pool(fet_2)
#flat_3=global_max_pool(fet_3)
#flat_4=global_max_pool(fet_4)
#flat_5=global_max_pool(fet_5)
#flat_6=global_max_pool(fet_6)
#flat_7=global_max_pool(fet_7)
#flat_8=global_max_pool(fet_8)

dens_1 = Dense(256, activation='relu', name='dens_1')(flat_1)
head_1 = Dense(1, activation='sigmoid', name='head_1')(dens_1)

#dens_2 = Dense(256, activation='relu', name='dens_2')(flat_2)
#head_2 = Dense(1, activation='sigmoid', name='head_2')(dens_2)

#dens_3 = Dense(256, activation='relu', name='dens_3')(flat_3)
#head_3 = Dense(1, activation='sigmoid', name='head_3')(dens_3)

#dens_4 = Dense(256, activation='relu', name='dens_4')(flat_4)
#head_4 = Dense(1, activation='sigmoid', name='head_4')(dens_4)

#dens_5 = Dense(256, activation='relu', name='dens_5')(flat_5)
#head_5 = Dense(1, activation='sigmoid', name='head_5')(dens_5)

#dens_6 = Dense(256, activation='relu', name='dens_6')(flat_6)
#head_6 = Dense(1, activation='sigmoid', name='head_6')(dens_6)

#dens_7 = Dense(256, activation='relu', name='dens_7')(flat_7)
#head_7 = Dense(1, activation='sigmoid', name='head_7')(dens_7)

#dens_8 = Dense(256, activation='relu', name='dens_8')(flat_8)
#head_8 = Dense(1, activation='sigmoid', name='head_8')(dens_8)

#model = tf.keras.models.Model([input_1,input_2,input_3,input_4,input_5,input_6,input_7], [head_1, head_2, head_3, head_4, head_5, head_6, head_7], name='multi-task_self-supervised')
model = tf.keras.models.Model(input_1, head_1, name='multi-task_self-supervised')

loss=[]
loss_weights=[]
for i in range(len(transformations)):
  loss.append('binary_crossentropy')
  loss_weights.append(1/len(transformations))

opt = keras.optimizers.Adam(learning_rate=0.0003)
model.compile(
    loss=loss,
    loss_weights=loss_weights,
    optimizer=opt,
    metrics=['accuracy']
)

model.summary()

class Logger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc=[]
        val_acc=[]
        for i in range(len(transformations)):
            acc.append(logs.get('accuracy'))
            val_acc.append(logs.get('val_accuracy'))
        print('='*30,epoch+1,'='*30)
        print('accuracy',acc)
        print("val_accuracy",val_acc)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.1,patience=5,restore_best_weights=True )
x_=[]
y_=[]
for i in range(len(transformations)):
    x_.append(x_train_pro[i])
    y_.append(y_train_pro[i])

history=model.fit(x_[0], y_[0], epochs=30 , validation_split=0.2, shuffle=True, callbacks=[Logger()], verbose=False)

model.save("multi-task_self-supervised")


fig = plt.figure(figsize=(len(transformations)*6,6))
for ii,title in zip(range(len(transformations)),transformations):
    ax = fig.add_subplot(1,7,ii+1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    ax.set_ylim([0.5,1.1])
plt.savefig('7 heads_n.jpeg')
plt.close(fig)

labled=np.load('best_data.npz')
x_L=labled['arr_0']
y_L=labled['arr_1']
x_L_val=labled['arr_2']
y_L_val=labled['arr_3']

from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Dropout

num_fet=3

input_C_ = Input(shape=(window_size,num_fet), name='input_C_')

fet_extrct=model.layers[1]

fet_extrct.trainable=False

fet_=fet_extrct(input_C_)

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