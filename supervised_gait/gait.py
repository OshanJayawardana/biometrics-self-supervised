import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten

from backbones import *
from data_load import *

frame_size = 50
#path = os.path.join(os.getcwd(), '..', 'gait_dataset', "IDNet's dataset", "user_coordinates")
path = r"gait_dataset\IDNet's dataset\user_coordinates"

x_train, y_train = data_loader(path, frame_size=50)
print(x_train.shape)
x_train = norma(x_train)

x_train, x_test, y_train, y_test = train_test_split(x_train, np.array(y_train), test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

ks = 3
con =4
num_classes=50
inputs = Input(shape=(50,3))
#x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same')(inputs)
#x = BatchNormalization()(x)
#x = ReLU()(x)
#x = MaxPooling1D(pool_size=4, strides=4)(x)
#x = Dropout(rate=0.1)(x)
#x = resnetblock(x, CR=32*con, KS=ks)
#x = resnetblock(x, CR=32*con, KS=ks)
#x = resnetblock(x, CR=64*con, KS=ks)
#x = resnetblock(x, CR=64*con, KS=ks)
#x = resnetblock(x, CR=128*con, KS=ks)
#x = resnetblock(x, CR=128*con, KS=ks)
#x = resnetblock_final(x, CR=128*con, KS=ks)
x = idnet(inputs)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)
resnettssd = Model(inputs, outputs)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=5)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.0001, decay_rate=0.5, decay_steps=1000000)# 0.0001, 0.9, 100000
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
resnettssd.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
history = resnettssd.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=callback, batch_size=128)

results = resnettssd.evaluate(x_test,y_test)
print("test acc:", results[1])

#Calculating kappa score
metric = tfa.metrics.CohenKappa(num_classes=num_classes, sparse_labels=True)
metric.update_state(y_true=y_test , y_pred=resnettssd.predict(x_test))
result = metric.result()
print('kappa score: ',result.numpy())