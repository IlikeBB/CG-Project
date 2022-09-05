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


leaky = LeakyReLU(alpha=0.3)
def threed_unet():
    inputs = Input((img_rows, img_cols, img_depth, 1))
    conv11 = Conv3D(32, (3, 3, 3), activation=leaky, padding='same')(inputs)
    conv11 = BatchNormalization(axis=4)(conv11)
    conc11 = concatenate([inputs, conv11], axis=4)
    conv12 = Conv3D(32, (3, 3, 3), activation=leaky, padding='same')(conc11)
    conv12 = BatchNormalization(axis=4)(conv12)
    conc12 = concatenate([inputs, conv12], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc12)

    conv21 = Conv3D(64, (3, 3, 3), activation=leaky, padding='same')(pool1)
    conv21 = BatchNormalization(axis=4)(conv21)
    conc21 = concatenate([pool1, conv21], axis=4)
    conv22 = Conv3D(64, (3, 3, 3), activation=leaky, padding='same')(conc21)
    conv22 = BatchNormalization(axis=4)(conv22)
    conc22 = concatenate([pool1, conv22], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc22)

    conv31 = Conv3D(128, (3, 3, 3), activation=leaky, padding='same')(pool2)
    conv31 = BatchNormalization(axis=4)(conv31)
    conc31 = concatenate([pool2, conv31], axis=4)
    conv32 = Conv3D(128, (3, 3, 3), activation=leaky, padding='same')(conc31)
    conv32 = BatchNormalization(axis=4)(conv32)
    conc32 = concatenate([pool2, conv32], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc32)

#     conv41 = Conv3D(256, (3, 3, 3), activation=leaky, padding='same')(pool3)
#     conv31 = BatchNormalization(axis=4)(conv31)
#     conc41 = concatenate([pool3, conv41], axis=4)
#     conv42 = Conv3D(256, (3, 3, 3), activation=leaky, padding='same')(conc41)
#     conv42 = BatchNormalization(axis=4)(conv42)
#     conc42 = concatenate([pool3, conv42], axis=4)
#     pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc42)

#     conv51 = Conv3D(512, (3, 3, 3), activation=leaky, padding='same')(pool4)
#     conc51 = concatenate([pool4, conv51], axis=4)
#     conv52 = Conv3D(512, (3, 3, 3), activation=leaky, padding='same')(conc51)
#     conc52 = concatenate([pool4, conv52], axis=4)
    
    conv51 = Conv3D(256, (3, 3, 3), activation=leaky, padding='same')(pool3)
    conc51 = concatenate([pool3, conv51], axis=4)
    conv52 = Conv3D(256, (3, 3, 3), activation=leaky, padding='same')(conc51)
    conc52 = concatenate([pool3, conv52], axis=4)

#     up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc52), conc42], axis=4)
#     conv61 = Conv3D(256, (3, 3, 3), activation=leaky, padding='same')(up6)
#     conc61 = concatenate([up6, conv61], axis=4)
#     conv62 = Conv3D(256, (3, 3, 3), activation=leaky, padding='same')(conc61)
#     conc62 = concatenate([up6, conv62], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc52), conv32], axis=4)
    conv71 = Conv3D(128, (3, 3, 3), activation=leaky, padding='same')(up7)
    conc71 = concatenate([up7, conv71], axis=4)
    conv72 = Conv3D(128, (3, 3, 3), activation=leaky, padding='same')(conc71)
    conc72 = concatenate([up7, conv72], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc72), conv22], axis=4)
    conv81 = Conv3D(64, (3, 3, 3), activation=leaky, padding='same')(up8)
    conc81 = concatenate([up8, conv81], axis=4)
    conv82 = Conv3D(64, (3, 3, 3), activation=leaky, padding='same')(conc81)
    conc82 = concatenate([up8, conv82], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc82), conv12], axis=4)
    conv91 = Conv3D(32, (3, 3, 3), activation=leaky, padding='same')(up9)
    conc91 = concatenate([up9, conv91], axis=4)
    conv92 = Conv3D(32, (3, 3, 3), activation=leaky, padding='same')(conc91)
    conc92 = concatenate([up9, conv92], axis=4)

    conv10 = Conv3D(3, (1, 1, 1), activation='softmax')(conc92)

    model = Model(inputs=[inputs], outputs=[conv10])

#     model.summary()
    #plot_model(model, to_file='model.png')

    return model

if __name__=="__main__":
    
    model=threed_unet()


# In[ ]:




