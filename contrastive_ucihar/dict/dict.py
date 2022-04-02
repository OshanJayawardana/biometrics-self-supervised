import os
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)  # for reproducibility

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Input
from keras import layers
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D, Conv1D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.constraints import non_neg
from tensorflow.keras.optimizers import RMSprop

plt.style.use('ggplot')

window_size=128
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

#x_train = np.load("eeg_data.npz")
X = np.copy(x_train)
print("Shape of the data: ", X.shape)

nb_epoch = 200

n_rows, n_cols = X.shape[1], X.shape[2]
print(n_rows, n_cols)

# nb_filters would be the number of dictionary atoms
nb_filters = 1024 # number of filters in Conv1D layers
nb_conv = 5 # size of the convolutional kernels in encoder
code_size = 5 # size of the convolutional kernels in decoder => this will be the size of the dictionary atoms

# using these values we will have 20 atoms with size of 5 => D \in R^{20 \times 5}

# winner takes all layer: only keep the highest value

def wtall(X):
    M = K.max(X, axis=(1), keepdims=True)
    R = K.switch(K.equal(X, M), X, K.zeros_like(X))
    return R

# encoder definition

enc = Sequential()
enc.add(Conv1D(nb_filters, (nb_conv), activation='relu', padding='same', input_shape=(n_rows, n_cols)))
enc.add(Conv1D(nb_filters, (nb_conv), activation='relu', padding='same')) 
pool_shape = enc.output_shape
enc.add(Lambda(function=wtall, output_shape=pool_shape[1:]))

enc.summary()

# decoder definition

dec = Sequential()
dec.add(Conv1D(3, (code_size), strides=1, padding='same',input_shape=pool_shape[1:], kernel_constraint=non_neg()))
#dec.add(Flatten())
dec.summary()

# Autoencoder with encoder and decoder

model = Sequential()
model.add(enc)
model.add(dec)

model.compile(loss='mae', optimizer='adam')
model.summary()

# training the model
history = model.fit(X, X, batch_size=32, epochs=nb_epoch, verbose=1 )

# Training curve
f1=plt.figure()
plt.plot(history.history['loss'])
plt.savefig('training.png')
plt.close(f1)


# The learned dictionaries

W = np.asarray(K.eval(dec.layers[0].weights[0]))


print(W.shape)
fig = plt.figure(figsize=(3,3))
for i in range(min(20,nb_filters)):
    plt.subplot(5,4,i+1)
    plt.plot(W[:,i,0])


fig.text(0.5, 0, 'Decoder Kernel Depth', ha='center')
fig.text(0, 0.5, 'Decoder Kernel Weights', va='center', rotation='vertical')
plt.savefig('kernels.png')

from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

# Create Negative Pairs
def negative_sample(x, enc, dec):
    
    enc_x0 = enc.predict(x.reshape(1,128,3))
    enc_x1 = np.zeros_like(enc_x0)
    
    std = np.std(enc_x0[enc_x0>0])
    
    ind0 = np.where(enc_x0 > std/2)
    ind1 = np.where(enc_x0 <= std/2)

    ind_select = np.zeros_like(ind0)
    for i in range(len(ind1)): 
        ind_tmp = ind1[i].copy()
        np.random.shuffle(ind_tmp)
        ind_tmp = ind_tmp[:len(ind0[0])]
        ind_select[i, :] = ind_tmp

    for j in range(len(ind0[0])):
        enc_x1[ind_select[0][j], ind_select[1][j], ind_select[2][j]] = enc_x0[ind0[0][j], ind0[1][j], ind0[2][j]]
    
    xr = dec.predict(enc_x1.reshape(1,128,-1)).reshape(128,3)
    
    xr = (xr-xr.min())/(xr.max()-xr.min())
    
    return xr, 1-x.reshape(1,128,3)


# Create Positive Pairs
def positive_sample(x, enc, dec):
    
    enc_x0 = enc.predict(x.reshape(1,128,3))
    
    enc_x1 = enc_x0
    std = np.std(enc_x0[enc_x0>0])
    
    mask = np.zeros_like(enc_x0)
    mask[np.ix_(*np.where(enc_x0>0))]=1
    
    noise = np.random.randn() * std/5 * mask
    xr = dec.predict(enc_x1 + np.abs(noise))
    xr1 = xr.reshape(128, 3)

    xr = dec.predict(enc_x0 * np.abs(enc_x0)>std)
    xr2 = xr.reshape(128, 3)
    
    xr1 = (xr1-xr1.min())/(xr1.max()-xr1.min())
    xr2 = (xr2-xr2.min())/(xr2.max()-xr2.min())
    
    return xr1, xr2


# Calculates the euclidean distance used in the output layer 
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    tf.cast(y_true, tf.float32)
    return K.mean(y_true * square_pred + (1.0 - y_true) * margin_square)


def create_pairs(x, enc, dec):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs1 = np.array([])
    pairs2 = np.array([])
    labels = []
    count=0
    for ind in range(x.shape[0]):

        
        xn1, xn2 = negative_sample(x[ind], enc, dec)
        xp1, xp2 = positive_sample(x[ind], enc, dec)
        
        xo=np.reshape(x[ind],(1,x[ind].shape[0],x[ind].shape[1]))
        #print('xn1',xn1.shape)
        xn1=np.reshape(xn1,(1,xn1.shape[0],xn1.shape[1]))
        #print('xn2',xn2.shape)
        #xn2=np.reshape(xn2,(1,xn2.shape[0],xn2.shape[1]))
        xp1=np.reshape(xp1,(1,xp1.shape[0],xp1.shape[1]))
        xp2=np.reshape(xp2,(1,xp2.shape[0],xp2.shape[1]))

        # similar pair
        
        if pairs1.shape[0]==0:
            pairs1=xo
        else:
            #print('pairs1',pairs1.shape)
            #print('xo',xo.shape)
            pairs1=np.concatenate((pairs1,xo),axis=0)

        if pairs2.shape[0]==0:
            pairs2=xp1
        else:
            pairs2=np.concatenate((pairs2,xp1),axis=0)
        
        # dissimilar pair
        pairs1=np.concatenate((pairs1,xo),axis=0)
        pairs2=np.concatenate((pairs2,xn1),axis=0)
        labels += [1, 0]
        
        pairs1=np.concatenate((pairs1,xo),axis=0)
        pairs2=np.concatenate((pairs2,xp2),axis=0)
        
        pairs1=np.concatenate((pairs1,xo),axis=0)
        pairs2=np.concatenate((pairs2,xn2),axis=0)

        labels += [1, 0]
        count+=1
        print(count)
        
    pairs1 = np.expand_dims(pairs1, axis=0)
    pairs2 = np.expand_dims(pairs2, axis=0)
    pairs = np.concatenate((pairs1,pairs2),axis=0)

        
    return np.reshape(pairs,(pairs.shape[1],pairs.shape[0],pairs.shape[2],pairs.shape[3])), np.array(labels,dtype='float32')


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(4, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)



def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

epochs = nb_epoch

enc = model.layers[0]
dec = model.layers[1]
enc.trainable=False
dec.trainable=False

# the data, split between train and test sets
#x_train = np.load("eeg_data.npz")
#x_train = x_train['arr_0']
x_train = x_train.astype('float32')

inds = np.arange(x_train.shape[0])
np.random.shuffle(inds)
np.random.shuffle(inds)
x_train = x_train[inds]

input_shape = x_train.shape[1:]
print('here',x_train.shape)

# create training+test positive and negative pairs
data_pairs, data_y = create_pairs(x_train, enc, dec)
print(data_pairs.shape)
print(data_y.shape)

print(data_y.shape)
inds_train_test = np.arange(data_y.shape[0])
print(inds_train_test)
np.random.shuffle(inds_train_test)
print(inds_train_test)
print(inds_train_test.shape)
data_pairs = data_pairs[inds_train_test]
data_y = data_y[inds_train_test]
tr_pairs = data_pairs[:5882]
te_pairs = data_pairs[5882:]

tr_y = data_y[:5882]
te_y = data_y[5882:]

print('tr_pairs', tr_pairs.shape)
print('te_pairs', te_pairs.shape)
print('tr_y', tr_y.shape)
print('te_y', te_y.shape)


# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=32,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))


y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

BATCH_SIZE = 256
EPOCHS = 300
# Create a cosine decay learning scheduler

data = np.load("best_data.npz")
X_ = data['arr_0']
Y_ = data['arr_1']

num_training_samples = len(X_)
print("num_training_samples",num_training_samples)
steps = EPOCHS * (num_training_samples // BATCH_SIZE)
print("steps",steps)

lr_decayed_fn = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.005, decay_steps=steps
)

backbone=model.layers[2]
backbone.trainable=False

inputs = Input((n_rows, n_cols))
x = backbone(inputs, training=False)
x = Dense(1024, activation="relu")(x)
outputs = Dense(15, activation="softmax")(x)
linear_model = Model(inputs, outputs, name="linear_model")

# Compile model and start training.
linear_model.compile(
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9),
)


history = linear_model.fit(X_
    , Y_,validation_data=(data['arr_2'],data['arr_3']), epochs=EPOCHS
)

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

_, test_acc = linear_model.evaluate(x_test, y_test)
print("Test accuracy: {:.2f}%".format(test_acc * 100))

plt.plot(history.history["loss"])
plt.grid()
plt.title("Classification loss")
plt.show()

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

#f2=plt.figure()
#plt.plot(history.history["loss"])
#plt.grid()
#plt.title("Classification loss")
#plt.show()