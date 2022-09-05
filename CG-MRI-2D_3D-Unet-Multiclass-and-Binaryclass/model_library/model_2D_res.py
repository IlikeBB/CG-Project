from keras.models import Model
from keras.layers import Input, concatenate
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
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
    conv12 = Conv2D(32, (3, 3), padding='same')(conv11)
    conv12 = BatchNormalization(axis=3)(conv12)
    conv12 = Activation(leaky)(conv12)
    conc12 = concatenate([inputs, conv12], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conc12)
    
    conv21 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv21 = BatchNormalization(axis=3)(conv21)
    conv21 = Activation(leaky)(conv21)
    conv22 = Conv2D(64, (3, 3), padding='same')(conv21)
    conv22 = BatchNormalization(axis=3)(conv22)
    conv22 = Activation(leaky)(conv22)
    conc22 = concatenate([pool1, conv22], axis=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conc22)

    conv31 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv31 = BatchNormalization(axis=3)(conv31)
    conv31 = Activation(leaky)(conv31)
    conv32 = Conv2D(128, (3, 3), padding='same')(conv31)
    conv32 = BatchNormalization(axis=3)(conv32)
    conv32 = Activation(leaky)(conv32)
    conc32 = concatenate([pool2, conv32], axis=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conc32)

    conv51 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv51 = BatchNormalization(axis=3)(conv51)
    conv51 = Activation(leaky)(conv51)
    conv52 = Conv2D(256, (3, 3), padding='same')(conv51)
    conv52 = Activation(leaky)(conv52)
    conv52 = BatchNormalization(axis=3)(conv52)
    conc52 = concatenate([pool3, conv52], axis=3)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc52), conv32], axis=3)
    conv71 = Conv2D(128, (3, 3), padding='same')(up7)
    conv71 = Activation(leaky)(conv71)
    conv72 = Conv2D(128, (3, 3), padding='same')(conv71)
    conv72 = Activation(leaky)(conv72)
    conc72 = concatenate([up7, conv72], axis=3)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc72), conv22], axis=3)
    conv81 = Conv2D(64, (3, 3), padding='same')(up8)
    conv81 = Activation(leaky)(conv81)
    conv82 = Conv2D(64, (3, 3), padding='same')(conv81)
    conv82 = Activation(leaky)(conv82)
    conc82 = concatenate([up8, conv82], axis=3)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc82), conv12], axis=3)
    conv91 = Conv2D(32, (3, 3), padding='same')(up9)
    conv91 = Activation(leaky)(conv91)
    conv92 = Conv2D(32, (3, 3), padding='same')(conv91)
    conv92 = Activation(leaky)(conv92)
    conc92 = concatenate([up9, conv92], axis=3)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conc92)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

