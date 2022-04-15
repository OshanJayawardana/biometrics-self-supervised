import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from sklearn.utils import class_weight

from backbones import *
from data_load import *
from losses import *

frame_size = 50 #130
num_classes=50
num_sessions=16
#path = os.path.join(os.getcwd(), '..', 'gait_dataset', "IDNet's dataset", "user_coordinates")
path = os.path.join(os.getcwd(), '..', 'gait_dataset', "idnet")
#path = os.path.join(os.getcwd(), '..', 'gait_dataset', "IDNet_dataset.csv")

#x_train, y_train = data_loader_csv(path, frame_size=frame_size, num_classes=num_classes)
x_train, y_train, weights = data_loader_8(path, frame_size=frame_size, num_classes=num_classes, num_sessions=num_sessions)
print(x_train.shape)
x_train = norma(x_train)

x_train, y_train, x_val, y_val, x_test, y_test = label_aware_split(x_train,y_train)
unique, counts = np.unique(y_train, return_counts=True)
weights = dict(zip(unique, (1/counts)/sum(1/counts)))

ks = 10
con =1
inputs = Input(shape=(frame_size,4))
#x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same')(inputs) 
#x = BatchNormalization()(x)
#x = ReLU()(x)
#x = MaxPooling1D(pool_size=4, strides=4)(x)
#x = Dropout(rate=0.1)(x)
#x = resnetblock(x, CR=32*con, KS=ks)
#x = resnetblock(x, CR=64*con, KS=ks)
#x = resnetblock(x, CR=128*con, KS=ks)
#x = resnetblock_final(x, CR=128*con, KS=ks)
x = lstm_model(inputs)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)
resnettssd = Model(inputs, outputs)
resnettssd.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=5)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.001, decay_rate=0.95, decay_steps=1000000)# 0.0001, 0.9, 100000
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
#optimizer = tf.keras.optimizers.Adam()
resnettssd.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
history = resnettssd.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=callback, batch_size=128, class_weight=weights)

results = resnettssd.evaluate(x_test,y_test)
print("test acc:", results[1])

#Calculating kappa score
metric = tfa.metrics.CohenKappa(num_classes=num_classes, sparse_labels=True)
metric.update_state(y_true=y_test , y_pred=resnettssd.predict(x_test))
result = metric.result()
print('kappa score: ',result.numpy())

inp = input("do you want to save the model? (y/) :")
if inp=="y":
  resnettssd.save(os.path.join("models", "gait_supervised_resnet"))