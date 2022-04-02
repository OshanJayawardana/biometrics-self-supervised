import numpy as np
import tensorflow as tf
import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D, Conv1D
from keras.constraints import non_neg

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

window_size=128
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

# hyperparameters
X = x_train
nb_epoch = 50
mlp_s = 2048
n_rows, n_cols = X.shape[1], X.shape[2]
frame_size,ftr = n_rows, n_cols
# nb_filters would be the number of dictionary atoms
nb_filters = 128 # number of filters in Conv1D layers
nb_conv = 32 # size of the convolutional kernels in encoder
code_size = 32 # size of the convolutional kernels in decoder => this will be the size of the dictionary atoms
samples_per_class = 100

def wtall(X):
    M = K.max(X, axis=(1), keepdims=True)
    R = K.switch(K.equal(X, M), X, K.zeros_like(X))
    return R

# encoder definition
enc = Sequential(name='enc')
enc.add(Conv1D(nb_filters, (nb_conv), activation='relu', padding='same', input_shape=(n_rows, n_cols)))
enc.add(Conv1D(nb_filters, (nb_conv), activation='relu', padding='same'))
pool_shape = enc.output_shape
enc.add(Lambda(function=wtall, output_shape=pool_shape[1:]))

enc.summary()

# decoder definition
dec = Sequential(name='dec')
dec.add(Conv1D(n_cols, (code_size), strides=1, padding='same',input_shape=pool_shape[1:], kernel_constraint=non_neg()))
#dec.add(Flatten())
dec.summary()

# Autoencoder with encoder and decoder
model = Sequential(name='dic')
model.add(enc)
model.add(dec)

model.compile(loss='mae', optimizer='adam')
model.summary()

# training the model
history = model.fit(X, X, batch_size=512, epochs=nb_epoch, verbose=1 )

enc0 = enc
enc0.trainable = False

dec0 = dec
dec0.trainable = False


#creating contrastive model
def positive_sample(x, enc, dec):
    enc_x0 = enc.predict(x.reshape(1,128,3))
    
    enc_x1 = enc_x0
    std = np.std(enc_x0[enc_x0>0])
    
    mask = np.zeros_like(enc_x0)
    mask[np.ix_(*np.where(enc_x0>0))]=1
    
    noise = np.random.randn() * std/5 * mask
    xr = dec.predict(enc_x1 + np.abs(noise))
    xr1 = xr.reshape(128,3)

    xr = dec.predict(enc_x0 * np.abs(enc_x0)>std)
    xr2 = xr.reshape(128,3)
    
    xr1 = (xr1-xr1.min())/(xr1.max()-xr1.min())
    xr2 = (xr2-xr2.min())/(xr2.max()-xr2.min())
    
    return xr1, xr2

def create_pairs(x, enc, dec):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs1 = np.array([])
    pairs2 = np.array([])
    labels = []

    for ind in range(x.shape[0]):
        xp1, xp2 = positive_sample(x[ind], enc, dec)
        xp1=np.reshape(xp1,(1,xp1.shape[0],xp1.shape[1]))
        xp2=np.reshape(xp2,(1,xp2.shape[0],xp2.shape[1]))

        # similar pair
        
        if pairs1.shape[0]==0:
            pairs1=xp1
        else:
            #print('pairs1',pairs1.shape)
            #print('xo',xo.shape)
            pairs1=np.concatenate((pairs1,xp1),axis=0)

        if pairs2.shape[0]==0:
            pairs2=xp2
        else:
            pairs2=np.concatenate((pairs2,xp2),axis=0)

    pairs1 = np.expand_dims(pairs1, axis=0)
    pairs2 = np.expand_dims(pairs2, axis=0)
    pairs = np.concatenate((pairs1,pairs2),axis=0)

        
    return pairs
    
x_con = np.load('dict_pairs.npz')['arr_0']
x_con = np.reshape(x_con,(x_con.shape[1],x_con.shape[0],x_con.shape[2],x_con.shape[3]))

def get_encoder():
    # Input and backbone.
    inputs = layers.Input((frame_size,ftr))
    x = layers.Conv1D(filters=32, kernel_size=24, activation='relu',kernel_regularizer=regularizers.l2(0.0001))(inputs)#64 #10
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters=64, kernel_size=16, activation='relu',kernel_regularizer=regularizers.l2(0.0001))(x)#128 #5
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters=96, kernel_size=8, activation='relu',kernel_regularizer=regularizers.l2(0.0001))(x)#128 #5
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
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
            layers.Dense(
                mlp_s//4,
                activation='relu',
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            layers.BatchNormalization(),
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
        ds_one, ds_two = data[0], data[1]

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

EPOCHS = 1000
# Create a cosine decay learning scheduler.
num_training_samples = x_con.shape[1]
print("num_training_samples",num_training_samples)
steps = EPOCHS * (num_training_samples // BATCH_SIZE)
print("steps",steps)
lr_decayed_fn = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.005, decay_steps=steps
)

# Create an early stopping callback.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=5, restore_best_weights=True
)

# Compile model and start training.
contrastive = Contrastive(get_encoder(), get_predictor())
contrastive.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.6))
history = contrastive.fit(x_con, epochs=EPOCHS, callbacks=[early_stopping], batch_size=BATCH_SIZE)

#loading data for classifier
data_clas = np.load('best_data.npz')
x_L = data_clas['arr_0']
y_L = data_clas['arr_1']
x_L_val = data_clas['arr_2']
y_L_val = data_clas['arr_3']

#training a classifier using encorder
#encoder = enc
backbone = tf.keras.Model(
    contrastive.encoder.input, contrastive.encoder.output
)

backbone.trainable=False
inputs = layers.Input((n_rows, n_cols))
x = backbone(inputs, training=False)
x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
outputs= layers.Dense(6, activation='softmax', name='outputs')(x)

clas = tf.keras.models.Model(inputs, outputs, name='classifier')
opt =  tf.keras.optimizers.Adam(learning_rate=0.0003)
clas.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history1 = clas.fit(x_L, y_L, validation_data=(x_L_val,y_L_val), batch_size=512, epochs=100, verbose=1 )

#evaluating results
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
results = clas.evaluate(x_test, y_test)
print("test acc:", results[1]*100,"%")

off=10
start=895
# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for "+str(off)+" samples")
predictions = clas.predict(x_test[start:start+off])
print("True labels: ",y_test[start:start+off])
y_pred_test=[]
for i in range(off):
    y_pred_test.append((np.where(predictions[i] == np.amax(predictions[i])))[0][0])
print("pred labels: ",np.array(y_pred_test))

#Calculating kappa score
metric = tfa.metrics.CohenKappa(num_classes=6, sparse_labels=True)
metric.update_state(y_true=y_test , y_pred=clas.predict(x_test))
result = metric.result()
print('kappa score: ',result.numpy())