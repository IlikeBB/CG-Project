from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

img_depth = 32
img_rows = 192
img_cols = 192
# leaky = LeakyReLU(alpha=0.3)
acti = LeakyReLU(alpha=0.3)

def conv_block(filter_n, kernel_n,input_,batch=True):
    input_ = Conv3D(filter_n, (kernel_n, kernel_n, kernel_n), padding='same')
    x = Activation(acti)(x)
    if batch==True:
        x = BatchNormalization(axis=4)(x)
    
    return batch
# conv1 = conv_block(32, 3, inputs, batch=True)

def get_unet():
    inputs = Input((img_depth, img_rows, img_cols, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation=acti, padding='same')(inputs)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = Conv3D(32, (3, 3, 3), activation=acti, padding='same')(conv1)
    conv1 = BatchNormalization(axis=4)(conv1)
    conc1 = concatenate([inputs, conv1], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc1)

    conv2 = Conv3D(64, (3, 3, 3), activation=acti, padding='same')(pool1)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = Conv3D(64, (3, 3, 3), activation=acti, padding='same')(conv2)
    conv2 = BatchNormalization(axis=4)(conv2)
    conc2 = concatenate([pool1, conv2], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc2)

    conv3 = Conv3D(128, (3, 3, 3), activation=acti, padding='same')(pool2)
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation=acti, padding='same')(conv3)
    conv3 = BatchNormalization(axis=4)(conv3)
    conc3 = concatenate([pool2, conv3], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc3)

#     conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
#     conc4 = concatenate([pool3, conv4], axis=4)
#     pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc4)

    conv4 = Conv3D(256, (3, 3, 3), activation=acti, padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation=acti, padding='same')(conv4)
    conc4 = concatenate([pool3, conv4], axis=4)

#     conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
#     conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
#     conc5 = concatenate([pool4, conv5], axis=4)

#     up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc5), conv4], axis=4)
#     conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
#     conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)
#     conc6 = concatenate([up6, conv6], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc4), conv3], axis=4)
    conv7 = Conv3D(128, (3, 3, 3), activation=acti, padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation=acti, padding='same')(conv7)
    conc7 = concatenate([up7, conv7], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc7), conv2], axis=4)
    conv8 = Conv3D(64, (3, 3, 3), activation=acti, padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation=acti, padding='same')(conv8)
    conc8 = concatenate([up8, conv8], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc8), conv1], axis=4)
    conv9 = Conv3D(32, (3, 3, 3), activation=acti, padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation=acti, padding='same')(conv9)
    conc9 = concatenate([up9, conv9], axis=4)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])


    return model
