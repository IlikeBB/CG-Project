from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

img_depth = 16
img_rows = 192
img_cols = 192
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def conv_block(inputs, filters):
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    return x

def threed_unet():
    inputs = Input((img_depth, img_rows, img_cols, 1))
    print(inputs)
    
    """Downsample"""
    conv1 = conv_block(inputs, 32)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = conv_block(pool1, 64)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = conv_block(pool2, 128)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = conv_block(pool3, 256)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    
    """bridge"""
    conv5 = conv_block(pool4, 512)
    
    """Upsample"""
    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), 
                                       strides=(2, 2, 2), padding='same')(conv5), conv4], axis=4)
    conv6 = conv_block(up6, 256)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), 
                                       strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
    conv7 = conv_block(up7, 128)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), 
                                       strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
    conv8 = conv_block(up8, 64)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), 
                                       strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
    conv9 = conv_block(up9, 32)
    
    """sigmoid"""
    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)


    model = Model(inputs=[inputs], outputs=[conv10])

    model.summary()
    #plot_model(model, to_file='model.png')

    model.compile(optimizer=Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, 
                                 epsilon=1e-08, decay=0.000000199), 
                  loss='binary_crossentropy', metrics=['accuracy',dice_coef])
    return model

if __name__=="__main__":
    
    model=threed_unet()


# In[ ]:




