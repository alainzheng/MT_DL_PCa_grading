# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:51:36 2021

@author: Alain
"""



import numpy as np
from skimage import io
import os
import time
import datetime
from create_data import load_train_data


tic = time.perf_counter()


histoImages, masks = load_train_data()
masksImages = masks[:,8]
saturImages = masks[:,9]

patchsize = 256
reso = 8


train = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/traindatabal.npy')
val = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/valdata.npy')


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

np.save( str('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/meanstd.npy'), mean_std_arr.astype(np.float32) )

toc = time.perf_counter()
print('toc-tic: ' + str(datetime.timedelta(seconds = toc-tic)))
