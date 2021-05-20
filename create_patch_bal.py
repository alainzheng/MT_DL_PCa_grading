# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:19:48 2021

@author: Alain
"""


import numpy as np
from skimage import io
import time
import datetime
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from create_data import load_train_data, create_images


histoImages, masks = load_train_data()
masksImages = masks[:,8]
saturImages = masks[:,9]

patchsize = 128
reso = 4
np.random.seed(19)

train = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/sd2traindata.npy')
# val = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/valdata.npy')

patch_adder = np.zeros((1,3))

for im in range(len(train)):
    ma = io.imread(masksImages[train[im, 0]])
    ma_patch = ma[train[im, 1]:train[im, 1]+patchsize*reso:reso, 
                  train[im, 2]:train[im, 2]+patchsize*reso:reso]
    
    if np.count_nonzero(ma_patch==0) > 0.8*patchsize**2:
        patch_adder = np.concatenate( ( patch_adder, train[im][np.newaxis,:] ) )
        if np.random.random() < 0.4:
            patch_adder = np.concatenate( ( patch_adder, train[im][np.newaxis,:] ) )

    if np.count_nonzero(ma_patch==1) > 0.8*patchsize**2:
        if np.random.random() < 0.7:
            patch_adder = np.concatenate( ( patch_adder, train[im][np.newaxis,:] ) )

patch_adder = patch_adder[1:]

new_train_array = np.concatenate( ( patch_adder, train ) )
np.random.shuffle(new_train_array) # returns nothing
np.save( str('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/sd2traindatabal.npy'), new_train_array.astype(np.uint32) )



trainbal = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/sd2traindatabal.npy')

# countbal = [0,0,0]
# for imnum in range(len(trainbal)):
#     ma = io.imread(masksImages[trainbal[imnum, 0]])
#     ma_patch = ma[trainbal[imnum, 1]:trainbal[imnum, 1]+patchsize*reso:reso, 
#                   trainbal[imnum, 2]:trainbal[imnum, 2]+patchsize*reso:reso]
#     countbal[0] +=np.count_nonzero(ma_patch==0)/patchsize**2
#     countbal[1] +=np.count_nonzero(ma_patch==1)/patchsize**2
#     countbal[2] +=np.count_nonzero(ma_patch==2)/patchsize**2

# print(countbal)


###### GET MEANSTD


train = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/sd2traindatabal.npy')


###############################
############################### mean and std
###############################

r_array = np.array([])
g_array = np.zeros([])
b_array = np.zeros([])

for imnum in range(len(train)):
    im = io.imread(histoImages[train[imnum, 0]])
    im_patch = im[train[imnum, 1]:train[imnum, 1]+patchsize*reso:reso, 
                  train[imnum, 2]:train[imnum, 2]+patchsize*reso:reso]
    r_array = np.append( r_array, im_patch[:,:,0].flatten() )
    g_array = np.append( g_array, im_patch[:,:,1].flatten() )
    b_array = np.append( b_array, im_patch[:,:,2].flatten() )

r_mean = r_array.mean()
g_mean = b_array.mean()
b_mean = g_array.mean()

r_std = r_array.std()
g_std = b_array.std()
b_std = g_array.std()

mean_std_arr = np.array([[r_mean, g_mean, b_mean],
                         [r_std, g_std, b_std]])


np.save( str('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/sd2meanstd.npy'), mean_std_arr.astype(np.float32) )
"""
############## VERIFY BAL

# count = [0,0,0]
# for imnum in range(len(train)):
#     ma = io.imread(masks[train[imnum, 0])
#     ma_patch = ma[train[imnum, 1]:train[imnum, 1]+patchsize*reso:reso, train[imnum, 2]:train[imnum, 2]+patchsize*reso:reso]
#     count[0] +=np.count_nonzero(ma_patch==0)/patchsize**2
#     count[1] +=np.count_nonzero(ma_patch==1)/patchsize**2
#     count[2] +=np.count_nonzero(ma_patch==2)/patchsize**2
    
# print(count)


"""