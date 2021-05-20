# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:00:18 2021

@author: Alain

try to output just a numpy array with (image number in correct list, x, y) of the patch
need to give index image number to histoimages -> order kept

first split in train and val, then put in same folder per patchsize and resolution
first this code then balance -> create_patch_bal

"""


import numpy as np
from skimage import io
import time
import datetime
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from create_data import load_train_data, load_test_data



############# test set

histoImages, masks = load_train_data()
# masksImages = masks[:,8]
saturImages = masks[:,9]

patchsize = 128
reso = 4
threshold = 0.7
seed = 10 #1 #19
validationSplit = 0.2
np.random.seed(seed)

imnumarray = np.arange(0,len(histoImages))   
saturtrain, saturval, imnumtrain, imnumval = train_test_split(saturImages, imnumarray, test_size=validationSplit, random_state=seed)


# selection_array = np.zeros((1,3))

# for imnum in range(len(saturImages)):
    
#     # preprocess mask
#     # ma = io.imread(str(masksImages[imnum]))
#     # im = io.imread(str(histoImages[imnum]))
#     sat = io.imread(str(saturImages[imnum]))
    
#     i0 = np.random.randint(0,patchsize*reso)
#     j0 = np.random.randint(0,patchsize*reso)
    
#     for i in range(i0, sat.shape[0], patchsize*reso):
#         for j in range(j0, sat.shape[1], patchsize*reso):
#             sat_patch = sat[i:i+patchsize*reso:reso,j:j+patchsize*reso:reso]
#             if sat_patch.shape[0] == patchsize and sat_patch.shape[1]==patchsize:
#                 if np.count_nonzero(sat_patch==1) >= patchsize**2 * threshold:
#                     selection_array = np.concatenate( ( selection_array, np.array([imnum, i, j])[np.newaxis,:] ) )

# selection_array = selection_array[1:]
# np.random.shuffle(selection_array) # returns nothing
# os.makedirs('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select', exist_ok=True)
# np.save( str('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/testdata.npy'), selection_array.astype(np.uint32) )





# ############# training

# histoImages, masks = load_train_data()
# # masksImages = masks[:,8]
# saturImages = masks[:,9]


# selection_array = np.zeros((1,3))

# for imnum in range(len(saturImages)):
    
#     # preprocess mask
#     # ma = io.imread(str(masksImages[imnum]))
#     # im = io.imread(str(histoImages[imnum]))
#     sat = io.imread(str(saturImages[imnum]))
    
#     i0 = np.random.randint(0,patchsize*reso)
#     j0 = np.random.randint(0,patchsize*reso)
    
#     for i in range(i0, sat.shape[0], patchsize*reso):
#         for j in range(j0, sat.shape[1], patchsize*reso):
#             sat_patch = sat[i:i+patchsize*reso:reso,j:j+patchsize*reso:reso]
#             if sat_patch.shape[0] == patchsize and sat_patch.shape[1]==patchsize:
#                 if np.count_nonzero(sat_patch==1) >= patchsize**2 * threshold:
#                     selection_array = np.concatenate( ( selection_array, np.array([imnum, i, j])[np.newaxis,:] ) )

# selection_array = selection_array[1:]
# np.random.shuffle(selection_array) # returns nothing
# os.makedirs('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select', exist_ok=True)
# np.save( str('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/fulltraindata.npy'), selection_array.astype(np.uint32) )







#### TRAIN VAL split

################
#### train data
################
selection_array = np.zeros((1,3))

for imnum in range(len(saturtrain)):
    
    # preprocess mask
    # ma = io.imread(str(masksImages[imnum]))
    # im = io.imread(str(histoImages[imnum]))
    sat = io.imread(str(saturtrain[imnum]))
    
    i0 = np.random.randint(0,patchsize*reso)
    j0 = np.random.randint(0,patchsize*reso)
    
    for i in range(i0, sat.shape[0], patchsize*reso):
        for j in range(j0, sat.shape[1], patchsize*reso):
            sat_patch = sat[i:i+patchsize*reso:reso,j:j+patchsize*reso:reso]
            if sat_patch.shape[0] == patchsize and sat_patch.shape[1]==patchsize:
                if np.count_nonzero(sat_patch==1) >= patchsize**2 * threshold:
                    selection_array = np.concatenate( ( selection_array, np.array([imnumtrain[imnum], i, j])[np.newaxis,:] ) )


selection_array = selection_array[1:]
np.random.shuffle(selection_array) # returns nothing
os.makedirs('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select', exist_ok=True)
np.save( str('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/sd2traindata.npy'), selection_array.astype(np.uint32) )

################
#### val data
################
selection_array = np.zeros((1,3))

for imnum in range(len(saturval)):
    
    # preprocess mask
    # ma = io.imread(str(masksImages[imnum]))
    # im = io.imread(str(histoImages[imnum]))
    sat = io.imread(str(saturval[imnum]))
    
    i0 = np.random.randint(0,patchsize*reso)
    j0 = np.random.randint(0,patchsize*reso)
    
    for i in range(i0, sat.shape[0], patchsize*reso):
        for j in range(j0, sat.shape[1], patchsize*reso):
            sat_patch = sat[i:i+patchsize*reso:reso,j:j+patchsize*reso:reso]
            if sat_patch.shape[0] == patchsize and sat_patch.shape[1]==patchsize:

                if np.count_nonzero(sat_patch==1) >= patchsize**2 * threshold:
                    selection_array = np.concatenate( ( selection_array, np.array([imnumval[imnum], i, j])[np.newaxis,:] ) )

selection_array = selection_array[1:]
np.random.shuffle(selection_array) # returns nothing
os.makedirs('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select', exist_ok=True)
np.save( str('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/sd2valdata.npy'), selection_array.astype(np.uint32) )





