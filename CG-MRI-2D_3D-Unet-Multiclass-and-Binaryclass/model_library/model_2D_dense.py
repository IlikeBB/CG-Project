#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from keras.losses import binary_crossentropy

img_depth = 1
img_rows = 192
img_cols = 192
smooth = 0.00001

leaky = LeakyReLU(alpha=0.3)
def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv11 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv11 = BatchNormalization(axis=3)(conv11)
    conv11 = Activation(leaky)(conv11)
    conc11 = concatenate([inputs, conv11], axis=3)
    conv12 = Conv2D(32, (3, 3), padding='same')(conc11)
    conv12 = BatchNormalization(axis=3)(conv12)
    conv12 = Activation(leaky)(conv12)
    conc12 = concatenate([inputs, conv12], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conc12)
    
    conv21 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv21 = BatchNormalization(axis=3)(conv21)
    conv21 = Activation(leaky)(conv21)
    conc21 = concatenate([pool1, conv21], axis=3)
    conv22 = Conv2D(64, (3, 3), padding='same')(conc21)
    conv22 = BatchNormalization(axis=3)(conv22)
    conv22 = Activation(leaky)(conv22)
    conc22 = concatenate([pool1, conv22], axis=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conc22)

    conv31 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv31 = BatchNormalization(axis=3)(conv31)
    conv31 = Activation(leaky)(conv31)
    conc31 = concatenate([pool2, conv31], axis=3)
    conv32 = Conv2D(128, (3, 3), padding='same')(conc31)
    conv32 = BatchNormalization(axis=3)(conv32)
    conv32 = Activation(leaky)(conv32)
    conc32 = concatenate([pool2, conv32], axis=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conc32)

#     conv41 = Conv2D(256, (3, 3), padding='same')(pool3)
#     conv41 = BatchNormalization(axis=3)(conv41)
#     conv41 = Activation(leaky)(conv41)
#     conc41 = concatenate([pool3, conv41], axis=3)
#     conv42 = Conv2D(256, (3, 3), padding='same')(conc41)
#     conv42 = BatchNormalization(axis=3)(conv42)
#     conv42 = Activation(leaky)(conv42)
#     conc42 = concatenate([pool3, conv42], axis=3)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conc42)

    conv51 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv51 = BatchNormalization(axis=3)(conv51)
    conv51 = Activation(leaky)(conv51)
    conc51 = concatenate([pool3, conv51], axis=3)
    conv52 = Conv2D(256, (3, 3), padding='same')(conc51)
    conv52 = Activation(leaky)(conv52)
    conv52 = BatchNormalization(axis=3)(conv52)
    conc52 = concatenate([pool3, conv52], axis=3)

#     conv51 = Conv2D(512, (3, 3), padding='same')(pool4)
#     conv51 = BatchNormalization(axis=3)(conv51)
#     conv51 = Activation(leaky)(conv51)
#     conc51 = concatenate([pool4, conv51], axis=3)
#     conv52 = Conv2D(512, (3, 3), padding='same')(conc51)
#     conv52 = Activation(leaky)(conv52)
#     conv52 = BatchNormalization(axis=3)(conv52)
#     conc52 = concatenate([pool4, conv52], axis=3)

#     up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc52), conc42], axis=3)
#     conv61 = Conv2D(256, (3, 3), padding='same')(up6)
#     conv61 = Activation(leaky)(conv61)
#     conc61 = concatenate([up6, conv61], axis=3)
#     conv62 = Conv2D(256, (3, 3), padding='same')(conc61)
#     conv62 = Activation(leaky)(conv62)
#     conc62 = concatenate([up6, conv62], axis=3)


    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc52), conv32], axis=3)
    conv71 = Conv2D(128, (3, 3), padding='same')(up7)
    conv71 = Activation(leaky)(conv71)
    conc71 = concatenate([up7, conv71], axis=3)
    conv72 = Conv2D(128, (3, 3), padding='same')(conc71)
    conv72 = Activation(leaky)(conv72)
    conc72 = concatenate([up7, conv72], axis=3)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc72), conv22], axis=3)
    conv81 = Conv2D(64, (3, 3), padding='same')(up8)
    conv81 = Activation(leaky)(conv81)
    conc81 = concatenate([up8, conv81], axis=3)
    conv82 = Conv2D(64, (3, 3), padding='same')(conc81)
    conv82 = Activation(leaky)(conv82)
    conc82 = concatenate([up8, conv82], axis=3)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc82), conv12], axis=3)
    conv91 = Conv2D(32, (3, 3), padding='same')(up9)
    conv91 = Activation(leaky)(conv91)
    conc91 = concatenate([up9, conv91], axis=3)
    conv92 = Conv2D(32, (3, 3), padding='same')(conc91)
    conv92 = Activation(leaky)(conv92)
    conc92 = concatenate([up9, conv92], axis=3)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conc92)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

