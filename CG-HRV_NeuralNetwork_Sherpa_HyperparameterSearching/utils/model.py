# define and fit the best model
import numpy as np,tensorflow as tf,keras,os
from matplotlib import pyplot
from keras import backend as K
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler,EarlyStopping
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers.normalization import BatchNormalization
from tensorflow_addons.optimizers import AdamW
from keras.optimizers import SGD,Adamax,RMSprop,Adam,Nadam
from sklearn.metrics import roc_auc_score,auc
from utils.visualize import train_history_visual
from datetime import datetime
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.python.keras.metrics import Metric
def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)


# def auc(y_true, y_pred):
#     auc = tf.metrics.auc(y_true, y_pred)[1]
#     K.get_session().run(tf.local_variables_initializer())
#     return auc

def best_model(train_X,test_X, train_Y,test_Y,input_shape,
               batch,epoch,lr,decay,momentum,af,
               units_L,dropout,units_num,warm_up):
    
# def best_model(train_X, train_Y,valida_X,valida_Y,test_X,test_Y,input_shape,batch,epoch,units,dropout,units_num,warm_up):
    #================================================ warm up =================================================== 
    class WarmupExponentialDecay(keras.callbacks.Callback):
        def __init__(self,lr_base=0.0002,lr_min=0.0,decay=0,warmup_epochs=0):
            self.num_passed_batchs = 0   #training epoch
            self.warmup_epochs=warmup_epochs  
            self.lr=lr_base #learning_rate_base
            self.lr_min=lr_min #nonuse
            self.decay=decay  #Index decay
            self.steps_per_epoch=0 #validation epoch

        def on_batch_begin(self, batch, logs=None):
            if self.steps_per_epoch==0:
                #Prevent it from being changed when running the validation set
                if self.params['steps'] == None:
                    self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
                else:
                    self.steps_per_epoch = self.params['steps']
            if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
                K.set_value(self.model.optimizer.lr,
                            self.lr*(self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
            else:
                K.set_value(self.model.optimizer.lr,
                            self.lr*((1-self.decay)**(self.num_passed_batchs-self.steps_per_epoch*self.warmup_epochs)))
            self.num_passed_batchs += 1
    #================================================ warm up ===================================================
    

        def on_epoch_begin(self,epoch,logs=None):
        #print current learning rate 
            print("learning_rate:%.10f"%(K.get_value(self.model.optimizer.lr))) 
            
            
            
    #================================================ LR reduce =================================================        
#     def scheduler(epoch):
#         # each 100 epochs,learning rate reduce to original 1/10
#         if epoch % 30 == 0 and epoch != 0:
#             lr = K.get_value(model.optimizer.lr)
#             K.set_value(model.optimizer.lr, lr * 0.1)
#             print("Learning Rate changed to {}".format(lr * 0.1))
#         return K.get_value(model.optimizer.lr)
    #================================================ LR reduce ================================================

    
    #================================================ main model ================================================
    model = Sequential()
    model.add(Dense(units_num, input_dim=input_shape,kernel_initializer='he_uniform'))

    model.add(BatchNormalization())

    model.add(Activation(af))
    model.add(Dropout(dropout))
    for _ in range(units_L):
        model.add(Dense(units_num,kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Activation(af))
        model.add(Dropout(dropout))

    model.add(Dense(units=1, activation='sigmoid'))

#     optimizer=RMSprop(lr=lr,decay=decay)
#     optimizer=AdamW(learning_rate=lr,weight_decay=decay)
#     optimizer=Nadam(lr=lr,schedule_decay=decay)
#     optimizer=Adam(learning_rate=lr,decay=decay)
    optimizer=Adamax(lr=lr, decay=decay)
#     optimizer = SGD(lr=lr, decay=decay,momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    #================================================ main model ================================================
    
    
    #================================================ call back =================================================

    callbacks_list=[EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='min')]

    #================================================ call back =================================================
    
    history=model.fit(train_X, train_Y, epochs=epoch, batch_size=batch, 
                      callbacks=callbacks_list,validation_split=0.1,
                      verbose=1,shuffle=True)    
#     history=model.fit(train_X, train_Y, epochs=epoch, batch_size=batch, 
#                       verbose=0)       
      
    train_history_visual(history)
              
    return model

