from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, Concatenate, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Dropout, GlobalMaxPooling1D, MaxPooling2D, Reshape

def incepblock(inputs, KS, CR):
    bottleneck = Conv1D(filters=CR*3 ,kernel_size=1,strides=1, activation='relu', use_bias=False, padding='same')(inputs)
    maxpool = MaxPooling1D(pool_size=3, strides=1, padding='same')(inputs)

    conv10 = Conv1D(filters=CR*3 ,kernel_size=KS, strides=1, activation='relu', use_bias=False, padding='same')(bottleneck)
    conv20 = Conv1D(filters=CR*3 ,kernel_size=KS+2, strides=1, activation='relu', use_bias=False, padding='same')(bottleneck)
    conv40 = Conv1D(filters=CR*3 ,kernel_size=KS+4, strides=1, activation='relu', use_bias=False, padding='same')(bottleneck)

    conv1 = Conv1D(filters=CR*3 ,kernel_size=1, strides=1, activation='relu', use_bias=False, padding='same')(maxpool)

    concat = Concatenate(axis=2)([conv10, conv20, conv40, conv1])
    norm = BatchNormalization()(concat)
    outputs = ReLU()(norm)
    outputs = MaxPooling1D(pool_size=4, strides=4)(concat)
    return outputs

def incepblock_final(inputs, KS, CR):
    bottleneck = Conv1D(filters=CR*3 ,kernel_size=1,strides=1, activation='relu', use_bias=False, padding='same')(inputs)
    maxpool = MaxPooling1D(pool_size=3, strides=1, padding='same')(inputs)

    conv10 = Conv1D(filters=CR*3 ,kernel_size=KS, strides=1, activation='relu', use_bias=False, padding='same')(bottleneck)
    conv20 = Conv1D(filters=CR*3 ,kernel_size=KS+2, strides=1, activation='relu', use_bias=False, padding='same')(bottleneck)
    conv40 = Conv1D(filters=CR*3 ,kernel_size=KS+4, strides=1, activation='relu', use_bias=False, padding='same')(bottleneck)

    conv1 = Conv1D(filters=CR*3 ,kernel_size=1, strides=1, activation='relu', use_bias=False, padding='same')(maxpool)

    concat = Concatenate(axis=2)([conv10, conv20, conv40, conv1])
    norm = BatchNormalization()(concat)
    outputs = ReLU()(norm)
    outputs = GlobalAveragePooling1D()(concat)
    return outputs

def shortcut(inputs, direct):
    short = Conv1D(filters=direct.shape[-1] ,kernel_size=1,strides=1, activation='relu', use_bias=False)(inputs)
    short = BatchNormalization(name='batchNormalization' )(short)

    sum_ = Add()([short, direct])
    outputs = ReLU()(sum_)

    return outputs

def resnetblock(inputs, KS, CR, skip=True):
    conv1 = Conv1D(filters=CR*3 ,kernel_size=KS, strides=1, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Conv1D(filters=CR*3 ,kernel_size=KS, strides=1, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    sum_ = Conv1D(filters=CR*3 ,kernel_size=KS, strides=1, padding='same')(conv2)

    if skip:
        conv4 = Conv1D(filters=CR*3 ,kernel_size=1, strides=1, padding='same')(inputs)
        sum_ = Add()([sum_, conv4])

    sum_ = BatchNormalization()(sum_)
    sum_ = ReLU()(sum_)
    #outputs = MaxPooling1D(pool_size=4, strides=4)(sum_)

    return sum_ 
    
def resnetblock_final(inputs, KS, CR, skip=True):
    conv1 = Conv1D(filters=CR*3 ,kernel_size=KS, strides=1, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Conv1D(filters=CR*3 ,kernel_size=KS, strides=1, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    sum_ = Conv1D(filters=CR*3 ,kernel_size=KS, strides=1, padding='same')(conv2)

    if skip:
        conv4 = Conv1D(filters=CR*3 ,kernel_size=1, strides=1, padding='same')(inputs)
        sum_ = Add()([sum_, conv4])

    sum_ = BatchNormalization()(sum_)
    sum_ = ReLU()(sum_)
    outputs = GlobalMaxPooling1D()(sum_)

    return outputs
    
def idnet(inputs):
    #best num_filters 3, 5, 7
    #best kernel sizes 8,8,8
    
    #new best num_filters 24, 40, 56
    #new best kernel sizes 8,8,8
    con=0.01
    x = Conv1D(filters=24, kernel_size=10, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(0.01*con))(inputs)#64 #10
    x = BatchNormalization()(x)
    x = ReLU()(x)
    #x = Dropout(0.35)(x)
    x = Conv1D(filters=40, kernel_size=10, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(0.01*con))(x)#64 #10
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    out_shape = x.shape
    x = Reshape(target_shape=(1,out_shape[1],out_shape[2]))(x)
    x = Conv2D(filters=56, kernel_size=(10,4), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(0.01*con))(x)#64 #10
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

    #x = Conv1DTranspose(filters=64, kernel_size=8)(x)
    #x = BatchNormalization()(x)
    #x = ReLU()(x)
    #x = Conv1DTranspose(filters=64, kernel_size=8)(x)
    #x = BatchNormalization()(x)
    #x = ReLU()(x)
    #x = Conv1DTranspose(filters=32, kernel_size=8)(x)
    #x = BatchNormalization()(x)
    #x = ReLU()(x)
    return x
    
def CAE_multi(inputs):
    #best num_filters 3, 5, 7
    #best kernel sizes 8,8,8
    
    #new best num_filters 24, 40, 56
    #new best kernel sizes 8,8,8
    con=1
    x = Conv1D(filters=24, kernel_size=8, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(0.01*con))(inputs)#64 #10
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    x = Dropout(0.35)(x)
    x = Conv1D(filters=40, kernel_size=8, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(0.01*con))(x)#64 #10
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    #x = Dropout(0.1)(x)
    x = Conv1D(filters=56, kernel_size=8, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(0.01*con))(x)#64 #10
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    #x = Dropout(0.05)(x)

    #x = Conv1DTranspose(filters=64, kernel_size=8)(x)
    #x = BatchNormalization()(x)
    #x = ReLU()(x)
    #x = Conv1DTranspose(filters=64, kernel_size=8)(x)
    #x = BatchNormalization()(x)
    #x = ReLU()(x)
    #x = Conv1DTranspose(filters=32, kernel_size=8)(x)
    #x = BatchNormalization()(x)
    #x = ReLU()(x)
    return x

def CAE_Origin(inputs):
    con=0.1
    x = Conv1D(filters=32, kernel_size=8, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(0.0001*con))(inputs)#64 #10
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    x = Dropout(0.35)(x)
    x = Conv1D(filters=64, kernel_size=8, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(0.0001*con))(x)#64 #10
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    x = Conv1D(filters=128, kernel_size=8, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(0.0001*con))(x)#64 #10
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    return x

#x = layers.Flatten()(inputs)
#x = layers.BatchNormalization()(x)
#x = layers.Dense(frame_size*ftr, kernel_regularizer=regularizers.l2(0.0001))(x)
#x = layers.Reshape((frame_size,ftr), input_shape=(frame_size*ftr,))(x)

#x = layers.Conv1D(filters=32, kernel_size=24,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(inputs)#64 #10
#x = layers.Dropout(0.1)(x)
#x = layers.Conv1D(filters=64, kernel_size=16,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(x)#128 #5
#x = layers.Dropout(0.1)(x)
#x = layers.Conv1D(filters=96, kernel_size=3,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(x)#128 #5
#x = layers.Dropout(0.1)(x)

#x = layers.MaxPooling1D(pool_size=2)(x)
#x = incepblock(inputs, KS=5, CR=8)
#direct = incepblock(x, KS=3, CR=16)
#direct = incepblock(direct, KS=3, CR=16)
#direct = incepblock(direct, KS=3, CR=16)
#x = shortcut(x,direct)
#direct = incepblock(x, KS=3, CR=32)
#x = incepblock_final(direct, KS=3, CR=32)

#x = resnetblock(inputs, KS=15, CR=32)
#x = resnetblock(x, KS=11, CR=64)
#x = resnetblock(x, KS=7, CR=128)
#x = resnetblock_final(x, KS=5, CR=128)

#x = GlobalAveragePooling1D()(direct)