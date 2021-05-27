# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:38:59 2021

@author: Alain
"""



# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 19:05:22 2020
@author: alain
"""

## common libraries
import numpy as np
from skimage import io
import os
import time
import datetime
from sklearn.model_selection import train_test_split
from operator import add

## tensorflow lib
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

## custom functions
from create_data import load_data, create_images
from unet import UNet
from get_augment import get_example
#from generatornNormalised import total_steps, val_generator, data_generator

                        

class SkMetrics(tf.keras.callbacks.Callback):
    
    def set_file_writer(self, tblogdir):
        self.file_writer = tf.summary.FileWriter(tblogdir)
    
    def on_train_begin(self, logs={}):
        print('\n training begin, log file in: ', tblogdir)

        #some initialisation
        self.rkVar = tf.Variable(0, dtype=tf.float32)
        self.rkSumm = tf.summary.scalar(name = 'Rk_statistic', tensor = self.rkVar)
        
        self.mccbeVar = tf.Variable(0, dtype=tf.float32)
        self.mccbeSumm = tf.summary.scalar(name = 'mcc_Benin', tensor = self.mccbeVar)
        self.mccgr3Var = tf.Variable(0, dtype=tf.float32)
        self.mccgr3Summ = tf.summary.scalar(name = 'mcc_grade3', tensor = self.mccgr3Var)
        self.mccgr45Var = tf.Variable(0, dtype=tf.float32)
        self.mccgr45Summ = tf.summary.scalar(name = 'mcc_grade4-5', tensor = self.mccgr45Var)        

        # f1score
        
        self.f1beVar = tf.Variable(0, dtype=tf.float32)
        self.f1beSumm = tf.summary.scalar(name = 'F1Beninscore', tensor = self.f1beVar)
        self.f1gr3Var = tf.Variable(0, dtype=tf.float32)
        self.f1gr3Summ = tf.summary.scalar(name = 'F1grade3score', tensor = self.f1gr3Var)
        self.f1gr45Var = tf.Variable(0, dtype=tf.float32)
        self.f1gr45Summ = tf.summary.scalar(name = 'F1grade45score', tensor = self.f1gr45Var)  


        TP = np.zeros(3); TN = np.zeros(3); FP = np.zeros(3); FN = np.zeros(3)
        # print('enter while code...')
        for valstep in range(val_per_epoch):
            Xval, true = next(self.validation_data)
            pred = np.asarray(model.predict(Xval))
            pred = to_categorical(np.argmax(pred, axis=-1), num_classes = 3)
            
            ### every single returns a [class 1, class 2, class 3]
            singleTP = np.sum( np.sum( true * pred, axis=(1,2) ), axis=0 )            # = TP
            singleTN = np.sum( np.sum( (1-true) * (1-pred), axis=(1,2) ), axis=0 )    # = TN

            singleFP = np.sum( np.sum( (1-true) * pred, axis=(1,2) ), axis=0 )        # = FP            
            singleFN = np.sum( np.sum( true * (1-pred), axis=(1,2) ), axis=0 )        # = FN
            

            ### the following are lists and not numpy arrays
            TP = TP + singleTP
            TN = TN + singleTN
            FP = FP + singleFP
            FN = FN + singleFN
                
        
        ### mcc per class
        numerator = (TP * TN - FP * FN)
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + np.finfo(float).eps
        mcc = numerator / denominator        
               
        ### Rk statistic
        tk= TP+FN         ## number of times class k truly occurred
        pk= TP+FP         ## number of times class k was predicted
        c = np.sum(TP)    ## total number of samples correctly predicted
        s = np.sum(TP+FN) ## total number of samples
        rknum = c*s-np.sum(tk*pk)
        rkden = np.sqrt(s**2-np.sum(pk*pk)) * np.sqrt(s**2-np.sum(tk*tk)) + np.finfo(float).eps
        rk = rknum / rkden
        
        sess.run(self.rkVar.assign(rk)) 
        self.file_writer.add_summary(sess.run(self.rkSumm), 0)
        
        sess.run(self.mccbeVar.assign(mcc[0])) 
        self.file_writer.add_summary(sess.run(self.mccbeSumm), 0)
        sess.run(self.mccgr3Var.assign(mcc[1])) 
        self.file_writer.add_summary(sess.run(self.mccgr3Summ), 0)
        sess.run(self.mccgr45Var.assign(mcc[2])) 
        self.file_writer.add_summary(sess.run(self.mccgr45Summ), 0)
        
        
        #f1score
        prec = TP / (TP + FP + np.finfo(float).eps)
        reca = TP / (TP + FN + np.finfo(float).eps)
        
        F1List = 2 * (prec * reca)/(prec + reca + np.finfo(float).eps)
        
        sess.run(self.f1beVar.assign(F1List[0])) 
        self.file_writer.add_summary(sess.run(self.f1beSumm), 0)
        sess.run(self.f1gr3Var.assign(F1List[1])) 
        self.file_writer.add_summary(sess.run(self.f1gr3Summ), 0)
        sess.run(self.f1gr45Var.assign(F1List[2])) 
        self.file_writer.add_summary(sess.run(self.f1gr45Summ), 0)
        
        self.file_writer.flush()


    def on_train_end(self, logs={}):
        print('training ended successfully, closing writer')
        self.file_writer.close() # not really needed, but good habit

    def on_epoch_end(self, epoch, logs={}):
        # keys = list(logs.keys())
        # print("End epoch {} of training; got log keys: {}".format(epoch, keys))    
    
        TP = np.zeros(3); TN = np.zeros(3); FP = np.zeros(3); FN = np.zeros(3)
        # print('enter while code...')
        for valstep in range(val_per_epoch):
            Xval, true = next(self.validation_data)
            pred = np.asarray(model.predict(Xval))
            pred = to_categorical(np.argmax(pred, axis=-1), num_classes = 3)
            
            ### every single returns a [class 1, class 2, class 3]
            singleTP = np.sum( np.sum( true * pred, axis=(1,2) ), axis=0 )            # = TP
            singleTN = np.sum( np.sum( (1-true) * (1-pred), axis=(1,2) ), axis=0 )    # = TN

            singleFP = np.sum( np.sum( (1-true) * pred, axis=(1,2) ), axis=0 )        # = FP            
            singleFN = np.sum( np.sum( true * (1-pred), axis=(1,2) ), axis=0 )        # = FN
            

            ### the following are lists and not numpy arrays
            TP = TP + singleTP
            TN = TN + singleTN
            FP = FP + singleFP
            FN = FN + singleFN
                
        
        ### mcc per class
        numerator = (TP * TN - FP * FN)
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + np.finfo(float).eps
        mcc = numerator / denominator        
               
        ### Rk statistic
        tk= TP+FN         ## number of times class k truly occurred
        pk= TP+FP         ## number of times class k was predicted
        c = np.sum(TP)    ## total number of samples correctly predicted
        s = np.sum(TP+FN) ## total number of samples
        rknum = c*s-np.sum(tk*pk)
        rkden = np.sqrt(s**2-np.sum(pk*pk)) * np.sqrt(s**2-np.sum(tk*tk)) + np.finfo(float).eps
        rk = rknum / rkden
        
        sess.run(self.rkVar.assign(rk)) 
        self.file_writer.add_summary(sess.run(self.rkSumm), epoch+1)
        
        sess.run(self.mccbeVar.assign(mcc[0])) 
        self.file_writer.add_summary(sess.run(self.mccbeSumm), epoch+1)
        sess.run(self.mccgr3Var.assign(mcc[1])) 
        self.file_writer.add_summary(sess.run(self.mccgr3Summ), epoch+1)
        sess.run(self.mccgr45Var.assign(mcc[2])) 
        self.file_writer.add_summary(sess.run(self.mccgr45Summ), epoch+1)
        
        
        #f1score
        prec = TP / (TP + FP + np.finfo(float).eps)
        reca = TP / (TP + FN + np.finfo(float).eps)
        
        F1List = 2 * (prec * reca)/(prec + reca + np.finfo(float).eps)
        
        sess.run(self.f1beVar.assign(F1List[0])) 
        self.file_writer.add_summary(sess.run(self.f1beSumm), epoch+1)
        sess.run(self.f1gr3Var.assign(F1List[1])) 
        self.file_writer.add_summary(sess.run(self.f1gr3Summ), epoch+1)
        sess.run(self.f1gr45Var.assign(F1List[2])) 
        self.file_writer.add_summary(sess.run(self.f1gr45Summ), epoch+1)
        
        self.file_writer.flush()

        

def val_generator(val, batchsize, patchsize):
    ### creates data for validation/testing
    ### function used to get validation set metrics on whole data set, use of batches to avoid 
    ### using too much memory
    while True:
        batchCounter = 0
        Xval = np.ndarray((batchsize,patchsize, patchsize, 3))# RGB image
        Yval = np.ndarray((batchsize,patchsize, patchsize, 3))# 3class ouTPut
        
        for imnum in range(len(val)):
            image = io.imread(str(histoImages[val[imnum, 0]])).astype(np.float32)
            image = (image-mean)/std ### standardisation
            target = io.imread(str(masksImages[val[imnum, 0]]))
            target = to_categorical(target, num_classes=3)

            Xval[batchCounter] = np.array(image[val[imnum, 1]:val[imnum, 1]+patchsize*reso:reso, 
                                                val[imnum, 2]:val[imnum, 2]+patchsize*reso:reso])
            Yval[batchCounter] = np.array(target[val[imnum, 1]:val[imnum, 1]+patchsize*reso:reso, 
                                                 val[imnum, 2]:val[imnum, 2]+patchsize*reso:reso])
            batchCounter += 1
            if batchCounter == batchsize:
                yield Xval, Yval
                batchCounter = 0
                            
        ### when the return shape is different than BS,PS,PS,3 -> ends the validation set
        ### metric calculation
        if batchCounter!=0:
            yield Xval[:batchCounter], Yval[:batchCounter]


def data_generator(train, batchsize, patchsize):
    ### creates data for training
    while True:
        ### some initialised data
        batchCounter = 0 # can also be before the while
        Xtrain = np.ndarray((batchsize,patchsize, patchsize,3))# RGB image
        Ytrain = np.ndarray((batchsize,patchsize, patchsize,3))# 3class ouTPut
        
        for imnum in range(len(train)):
            image = io.imread(str(histoImages[train[imnum, 0]])).astype(np.float32)
            image = (image-mean)/std ### standardisation
            
            ####### CHANGE HERE FOR SWITCHING BETWEEN EXPERT AND STAPLE ANALYSIS

            ####### EXPERTS analysis here
            # expert = np.random.choice(np.array(np.where(masks[train[imnum, 0], 0:6]!='NoGT'))[0])
            # target = io.imread(str(masks[train[imnum, 0], expert]))
            # target[target <= 2] = 0; target[target == 3] = 1; target[target >= 4] = 2
            # target = to_categorical(target, num_classes=3)
            
            ####### STAPLE analysis here
            target = io.imread(str(masksImages[train[imnum, 0]]))
            target = to_categorical(target, num_classes=3)
            
            img, tar = get_example(image[train[imnum, 1]:train[imnum, 1]+patchsize*reso:reso, 
                                         train[imnum, 2]:train[imnum, 2]+patchsize*reso:reso],
                                   target[train[imnum, 1]:train[imnum, 1]+patchsize*reso:reso, 
                                         train[imnum, 2]:train[imnum, 2]+patchsize*reso:reso],
                                   prob = 0.8)
            Xtrain[batchCounter] = img
            Ytrain[batchCounter] = tar
            
            batchCounter += 1
            if batchCounter == batchsize:
                yield Xtrain, Ytrain
                batchCounter = 0



if __name__ == '__main__':

    tic = time.perf_counter()

    ## remove warnings, helps readibility
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print('-'*30)
    print('running full training set train.py ...')
    print('-'*30)
    
    ########################################
    ##########   LOAD DATA    ##############
    ########################################
    
    histoImages, masks = load_data()
    masksImages = masks[:,8] # 3 class staple


    ########################################
    ###   TRAINING & VALIDATION SETS    ####
    ########################################
    ########################################
    ##########   PARAMETERS   ##############
    ######################################## 

    batchsize = 20
    patchsize = 128 # for patch size try 512, 256
    reso = 4
        
    train = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/fulltraindatabal1.npy')
    val = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/testdata.npy')
    val[:,0] = val[:,0] + 167     
    [mean, std] = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/meanstd.npy')    

    Steps_per_epoch = np.ceil(len(train)/batchsize).astype(np.uint16) ### 27318 for 
    val_per_epoch = np.ceil(len(val)/batchsize).astype(np.uint16)
    features = 48
    blocks = 5
    adam = 0.0001
    training_name = 'testing2/pc16exp'
    epoch_folder_name = 'epochs/'+training_name+'/'
    tblogdir = './logs/'+training_name+'{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    
    np.random.seed(19)
    ########################################
    #########   MODEL & SESSION   ##########
    ########################################

    tf.reset_default_graph()   
    sess = tf.Session()
    graph = tf.get_default_graph()
    set_session(sess)
    
    model = UNet((patchsize,patchsize,3), filters=features, blocks = blocks)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(adam),
                  metrics=['accuracy'])
    
    
    ########################################
    #############  CALLBACKS   #############
    ########################################
    
    os.makedirs(epoch_folder_name, exist_ok=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                            log_dir=tblogdir,
                            write_graph=False)
    tensorboard_callback.set_model(model)
    skmetrics = SkMetrics()
    skmetrics.set_file_writer(tblogdir)
    
    Callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, mode='min'),
            ModelCheckpoint(epoch_folder_name+'best_weights{epoch:02d}.h5',
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=True,
                        mode='min'),
            ModelCheckpoint(epoch_folder_name+'ep{epoch:02d}.h5',
                        save_weights_only=True,
                        period=1),
            tensorboard_callback,  ###tensorboard event will be in logs\training_name folder    
            skmetrics]

    ########################################
    #############  TRAINING    #############
    ########################################

    
    print('-'*30)
    print('fitting model')
    print('-'*30)
    
    history = model.fit_generator(data_generator(train, batchsize, patchsize),
                        steps_per_epoch=Steps_per_epoch,
                        epochs=100,
                        verbose=2,
                        callbacks=Callbacks,
                        validation_data=val_generator(val, batchsize, patchsize),
                        validation_steps=val_per_epoch,
                        shuffle=False
                        )
    

    ########################################
    #############    TIMER     #############
    ########################################
    
    toc = time.perf_counter()
    print('toc-tic: ' + str(datetime.timedelta(seconds = toc-tic)))
      
