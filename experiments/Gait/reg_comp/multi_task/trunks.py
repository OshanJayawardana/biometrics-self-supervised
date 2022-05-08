from backbones import *
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import regularizers

def trunk_1(frame_size, ft_len, reg_con): # 16
    con=1
    ks=3
    input_ = Input(shape=(frame_size, ft_len), name='input_')
    x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same',kernel_regularizer=regularizers.L2(reg_con))(input_) 
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(rate=0.1)(x)
    x = Flatten()(x)
    
    return tf.keras.models.Model(input_,x,name='trunk_')

def trunk_2(frame_size, ft_len, reg_con): # 16, 32
    con=1
    ks=3
    input_ = Input(shape=(frame_size, ft_len), name='input_')
    x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same',kernel_regularizer=regularizers.L2(reg_con))(input_) 
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(rate=0.1)(x)
    x = resnetblock_final(x, CR=32*con, KS=ks, reg_con=reg_con)
    
    return tf.keras.models.Model(input_,x,name='trunk_')
    
def trunk_3(frame_size, ft_len, reg_con): # 48, 96
    con=3
    ks=3
    input_ = Input(shape=(frame_size, ft_len), name='input_')
    x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same',kernel_regularizer=regularizers.L2(reg_con))(input_) 
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(rate=0.1)(x)
    x = resnetblock_final(x, CR=32*con, KS=ks, reg_con=reg_con)
    
    return tf.keras.models.Model(input_,x,name='trunk_')
    
def trunk_4(frame_size, ft_len, reg_con): # 64, 128
    con=4
    ks=3
    input_ = Input(shape=(frame_size, ft_len), name='input_')
    x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same',kernel_regularizer=regularizers.L2(reg_con))(input_) 
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(rate=0.1)(x)
    x = resnetblock_final(x, CR=32*con, KS=ks, reg_con=reg_con)
    
    return tf.keras.models.Model(input_,x,name='trunk_')
    
def trunk_5(frame_size, ft_len, reg_con): # 16, 32, 64
    con=1
    ks=3
    input_ = Input(shape=(frame_size, ft_len), name='input_')
    x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same',kernel_regularizer=regularizers.L2(reg_con))(input_) 
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(rate=0.1)(x)
    x = resnetblock(x, CR=32*con, KS=ks)
    x = resnetblock_final(x, CR=64*con, KS=ks, reg_con=reg_con)
    
    return tf.keras.models.Model(input_,x,name='trunk_')
    
def trunk_6(frame_size, ft_len, reg_con): # 48, 96, 192
    con=3
    ks=3
    input_ = Input(shape=(frame_size, ft_len), name='input_')
    x = Conv1D(filters=16*con,kernel_size=ks,strides=1, padding='same',kernel_regularizer=regularizers.L2(reg_con))(input_) 
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(rate=0.1)(x)
    x = resnetblock(x, CR=32*con, KS=ks)
    x = resnetblock_final(x, CR=64*con, KS=ks, reg_con=reg_con)
    
    return tf.keras.models.Model(input_,x,name='trunk_')