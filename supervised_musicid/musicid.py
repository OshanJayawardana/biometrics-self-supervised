import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers
from sklearn.manifold import TSNE

from backbones import *
from data_loader import *

frame_size   = 30
num_classes  = 20
path = "data"

x_train, y_train = data_load(path, frame_size = frame_size)

x_train = norma(x_train)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

ks = 3
con =1
inputs = Input(shape=(frame_size, x_train.shape[-1]))
x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same')(inputs) 
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling1D(pool_size=4, strides=4)(x)
x = Dropout(rate=0.1)(x)
# = resnetblock(x, CR=32*con, KS=ks)
#x = resnetblock(x, CR=64*con, KS=ks)
#x = resnetblock(x, CR=128*con, KS=ks)
#x = resnetblock_final(x, CR=128*con, KS=ks)
#x = lstm_model(inputs)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)
resnettssd = Model(inputs, outputs)
resnettssd.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=5)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.001, decay_rate=0.95, decay_steps=1000000)# 0.0001, 0.9, 100000
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
#optimizer = tf.keras.optimizers.Adam()
resnettssd.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
history = resnettssd.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=callback, batch_size=8)

x_train, y_train, x_val, y_val = 0, 0, 0, 0

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

enc_results = resnettssd(x_test)
enc_results = np.array(enc_results)
X_embedded = TSNE(n_components=2).fit_transform(enc_results)
fig4 = plt.figure(figsize=(18,12))
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_test)
plt.legend()
plt.savefig('graphs/latentspace_'+str(num_classes)+'.png')
plt.close(fig4)
#plt.show()
x_test, y_test = 0, 0

inp = input("do you want to save the model? (y/) :")
if inp=="y":
  resnettssd.save(os.path.join("models", "mindid_supervised"))