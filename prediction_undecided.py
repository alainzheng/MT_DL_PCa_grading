# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:43:34 2021



get rk per classification -> not pixel

@author: Alain
"""

# import tensorflow as tf
import numpy as np
from skimage import io
import time
import datetime
import matplotlib.pyplot as plt
import os
from unet import UNet

import pandas as pd
import seaborn as sns
from create_data import load_data
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix
from skimage.measure import label, regionprops



def prediction_dontknow(imdata, imlist, dirs_staexp):
    
    offset = 0
    for imnum in imlist:

        im = (io.imread(str(histoImages[imnum]))-mean)/std
        
        count = np.count_nonzero(imdata[:,0]==imnum)
        data_array = imdata[offset : offset + count]
        offset += count
        
        
        #############################
        ## make a numpy array based on the position of the patches
        histo_array = np.zeros((count, patchsize, patchsize, 3))
        for k in range(len(data_array)):
            histo_array[k] = np.array(im[data_array[k, 1]:data_array[k, 1]+patchsize*reso:reso, 
                                         data_array[k, 2]:data_array[k, 2]+patchsize*reso:reso])
        
        #############################
        ## predict on a sigle image (input as patches of the image)
        unet_pred = model.predict(histo_array, verbose=1)
        
        ## transform one-hot encoding into array of shape (ps,ps) with value = class

        resind = np.zeros((patchsize, patchsize))
        resmax = np.zeros((patchsize, patchsize))
        result = np.ones((count, patchsize, patchsize))*3
        pred_map = np.zeros((im.shape[0:2]))
        
        for k in range(len(data_array)):
            
            
            resmax = np.amax(unet_pred[k], axis=-1)
            resind = np.argmax(unet_pred[k], axis=-1)
            res2max = np.sort(unet_pred[k], axis=-1)[:,:,1] # second biggest value
            
            result[k][resmax > 2*res2max] = resind[resmax > 2*res2max]
            pred_map[data_array[k, 1]:data_array[k, 1]+patchsize*reso:reso, 
                     data_array[k, 2]:data_array[k, 2]+patchsize*reso:reso] = result[k]
        
        # i0 = data_array[0, 1]%reso
        # j0 = data_array[0, 2]%reso
        # io.imshow(pred_map[i0::reso,j0::reso])
        # io.show()


        #############################
        ##### unet prediction
    
        imageIndex = str(histoImages[imnum].replace('.jpg', '').replace('Data/Train Imgs/', ''))
        os.makedirs('Unets/'+name+'/' + dirs_staexp + '/', exist_ok=True)
        io.imsave(str('Unets/'+name+'/' + dirs_staexp + '/' +str(imageIndex)+'.png'), pred_map.astype(np.uint8), check_contrast= False)

        # os.makedirs('UnetsTest/' + dirs_staexp + '/', exist_ok=True)
        # io.imsave(str('UnetsTest/' + dirs_staexp + '/' +str(imageIndex)+'.png'), pred_map.astype(np.uint8), check_contrast= False)




def prediction_dontknow0(imdata, imlist, dirs_staexp):
    
    offset = 0
    for imnum in imlist:

        im = (io.imread(str(histoImages[imnum]))-mean)/std
        
        count = np.count_nonzero(imdata[:,0]==imnum)
        data_array = imdata[offset : offset + count]
        offset += count
        
        
        #############################
        ## make a numpy array based on the position of the patches
        histo_array = np.zeros((count, patchsize, patchsize, 3))
        for k in range(len(data_array)):
            histo_array[k] = np.array(im[data_array[k, 1]:data_array[k, 1]+patchsize*reso:reso, 
                                         data_array[k, 2]:data_array[k, 2]+patchsize*reso:reso])
        
        #############################
        ## predict on a sigle image (input as patches of the image)
        unet_pred = model.predict(histo_array, verbose=1)
        
        ## transform one-hot encoding into array of shape (ps,ps) with value = class

        resind = np.zeros((patchsize, patchsize))
        resmax = np.zeros((patchsize, patchsize))
        result = np.ones((count, patchsize, patchsize))*3
        pred_map = np.zeros((im.shape[0:2]))
        
        for k in range(len(data_array)):
            
            
            resmax = np.amax(unet_pred[k], axis=-1)
            resind = np.argmax(unet_pred[k], axis=-1)
            res2max = np.sort(unet_pred[k], axis=-1)[:,:,1] # second biggest value
            
            result[k][resmax > 0*res2max] = resind[resmax > 0*res2max]
            pred_map[data_array[k, 1]:data_array[k, 1]+patchsize*reso:reso, 
                     data_array[k, 2]:data_array[k, 2]+patchsize*reso:reso] = result[k]
        
        # i0 = data_array[0, 1]%reso
        # j0 = data_array[0, 2]%reso
        # io.imshow(pred_map[i0::reso,j0::reso])
        # io.show()


        #############################
        ##### unet prediction
    
        imageIndex = str(histoImages[imnum].replace('.jpg', '').replace('Data/Train Imgs/', ''))
        os.makedirs('Unets/'+name+'/' + dirs_staexp + '/', exist_ok=True)
        io.imsave(str('Unets/'+name+'/' + dirs_staexp + '/' +str(imageIndex)+'.png'), pred_map.astype(np.uint8), check_contrast= False)

        # os.makedirs('UnetsTest/' + dirs_staexp + '/', exist_ok=True)
        # io.imsave(str('UnetsTest/' + dirs_staexp + '/' +str(imageIndex)+'.png'), pred_map.astype(np.uint8), check_contrast= False)



if __name__ == '__main__':

    
    tic = time.perf_counter()

    histoImages, masks = load_data()
    
    # network
    patchsize = 128
    reso = 4
    features = 48
    blocks = 5

    # loading
    [mean, std] = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/meanstd.npy')    
    train = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/traindatabal.npy')
    val = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/valdata.npy')
    val[:,0] = val[:,0]
    train = train[train[:, 0].argsort()]    
    val = val[val[:, 0].argsort()]
    imnumtrain = np.unique(train[:,0])
    imnumval = np.unique(val[:,0])
        
    model = UNet((patchsize,patchsize,3), filters=features, blocks=blocks) 

    ############################# 0   1
    ############################# 13  16
    
    train_num = [0,0,1, 1]    ### used to choose the training 
    epoch_num = [13,14, 16, 17]    ### used to choose the weight in the training (epoch)
    dirs = os.listdir('epochs/validation2')
    """
    #############################
    ##    compute score confusion matrix
    
    for em in range(4):
        dirs2 = os.listdir('epochs/validation2/'+dirs[train_num[em]])
        dirs_staexp = dirs[train_num[em]]+dirs2[epoch_num[em]].replace('.h5', '') +'/'
        weights = 'epochs/validation2/'+dirs[train_num[em]] +'/'+ dirs2[epoch_num[em]]
        model.load_weights(weights)
        name = "val2_20"
        prediction_dontknow(val, imnumval, dirs_staexp)
        name = "val2_00"
        prediction_dontknow0(val, imnumval, dirs_staexp)


    #############################
    ##    validation3
    val = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/sd1valdata.npy')
    val[:,0] = val[:,0]
    val = val[val[:, 0].argsort()]
    imnumval = np.unique(val[:,0])
    
    dirs = os.listdir('epochs/validation3')
    
    
    for em in range(4):
        dirs2 = os.listdir('epochs/validation3/'+dirs[train_num[em]])
        dirs_staexp = dirs[train_num[em]]+dirs2[epoch_num[em]].replace('.h5', '') +'/'
        weights = 'epochs/validation3/'+dirs[train_num[em]] +'/'+ dirs2[epoch_num[em]]
        model.load_weights(weights)
        name = "val3_20"
        prediction_dontknow(val, imnumval, dirs_staexp)
        name = "val3_00"
        prediction_dontknow0(val, imnumval, dirs_staexp)
    """
    # for em in range(2):
    #     dirs2 = os.listdir('epochs/validation2/'+dirs[train_num[em]])
    #     dirs_staexp = dirs[train_num[em]]+dirs2[epoch_num[em]].replace('.h5', '') +'/'
    #     weights = 'epochs/validation2/'+dirs[train_num[em]] +'/'+ dirs2[epoch_num[em]].replace('.h5', '')
    #     model.load_weights(weights)
    #     prediction_dontknow(train, imnumtrain, dirs_staexp)

    #############################
    ##########  testing  ########
    #############################
    
    
    
    val = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/testdata.npy')
    val[:,0] = val[:,0] + 167    
    val = val[val[:, 0].argsort()]
    name = "test2_20"
    imnumval = np.unique(val[:,0])
        
    dirs = os.listdir('epochs/testing2')
    
    #############################
    ##    compute score confusion matrix

    
    for em in range(4):
        dirs2 = os.listdir('epochs/testing2/'+dirs[train_num[em]])
        dirs_staexp = dirs[train_num[em]]+dirs2[epoch_num[em]].replace('.h5', '') +'/'
        weights = 'epochs/testing2/'+dirs[train_num[em]] +'/'+ dirs2[epoch_num[em]]
        model.load_weights(weights)
        name = "test2_20"
        prediction_dontknow(val, imnumval, dirs_staexp)
        name = "test2_00"
        prediction_dontknow0(val, imnumval, dirs_staexp)

    
    # for em in range(2):
    #     dirs2 = os.listdir('epochs/testing2/'+dirs[train_num[em]])
    #     dirs_staexp = dirs[train_num[em]]+dirs2[epoch_num[em]].replace('.h5', '') +'/'
    #     weights = 'epochs/testing2/'+dirs[train_num[em]] +'/'+ dirs2[epoch_num[em]].replace('.h5', '')
    #     model.load_weights(weights)
    #     prediction_dontknow(train, imnumtrain, dirs_staexp)

    
    
    ########################################
    #############    TIMER     #############
    ########################################
    
    toc = time.perf_counter()
    print('toc-tic: ' + str(datetime.timedelta(seconds = toc-tic)))
      

    