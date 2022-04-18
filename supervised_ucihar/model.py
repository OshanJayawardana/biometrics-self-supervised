import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Concatenate, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from backbones import *
from data_load import *
from transformations import *

isall=True
domain = 0
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128

if domain:
    num_classes=6
else:
    num_classes=30

if isall:
    num_samp = ["Placeholder"]
else:
    num_samp = [5,10,15,20,25,35,45,55,65,75,95,115,135,155,175,215,255,295,335,375,455,535,615,695,775,935,1000]

x_train, y_train = data_fetch(norma=True, filt=False, filt_sigma=5, mix=True, domain=domain)
x_train = norma(x_train)
#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4)#0.4
#x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
x_train, y_train, x_val, y_val, x_test, y_test = label_aware_split(x_train,y_train)

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
  
SEED = 34
ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
#ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
ssl_ds_one = (
    ssl_ds_one.shuffle(1024, seed=SEED)
    .map(tf_aug2, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

ssl_ds_two = tf.data.Dataset.from_tensor_slices(y_train)
ssl_ds_two = (
    ssl_ds_two.shuffle(1024, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))

val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val = val.batch(BATCH_SIZE).prefetch(AUTO)

kappa_arr=np.array([])
acc_arr=np.array([])
for samples_per_class in num_samp:
    kappa=[]
    test_ac=[]
    for count in range(1):
        ks = 3
        con =1
        inputs = Input(shape=(128,4))
        x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling1D(pool_size=4, strides=4)(x)
        x = Dropout(rate=0.1)(x)
        x = resnetblock(x, CR=32*con, KS=ks)
        x = resnetblock(x, CR=64*con, KS=ks)
        x = resnetblock(x, CR=128*con, KS=ks)
        x = resnetblock_final(x, CR=128*con, KS=ks)
        #x = lstm_model(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        resnettssd = Model(inputs, outputs)
        resnettssd.summary()
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=5)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.001, decay_rate=0.95, decay_steps=1000000)# 0.0001, 0.9, 100000
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        resnettssd.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
        history = resnettssd.fit(x_train,y_train, validation_data=(x_val, y_val), epochs=100, callbacks=callback, batch_size=128)
        
        results = resnettssd.evaluate(x_test,y_test)
        print("test acc:", results[1])
        
        #Calculating kappa score
        metric = tfa.metrics.CohenKappa(num_classes=num_classes, sparse_labels=True)
        metric.update_state(y_true=y_test , y_pred=resnettssd.predict(x_test))
        result = metric.result()
        print('kappa score: ',result.numpy())
        
        resnettssd = tf.keras.Model(
            resnettssd.input, resnettssd.layers[-4].output
        )
        resnettssd.summary()
        
        enc_results = resnettssd(x_train)
        enc_results = np.array(enc_results)
        X_embedded = TSNE(n_components=2).fit_transform(enc_results)
        fig4 = plt.figure(figsize=(18,12))
        plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_train, label=y_train)
        plt.legend()
        plt.savefig('graphs/latentspace_'+str(num_classes)+'.png')
        plt.close(fig4)
        
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
