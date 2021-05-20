# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:29:13 2021

@author: Alain
"""


import numpy as np
from skimage import io
import time
import datetime
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from create_data import load_train_data, load_data


histoImages, masks = load_data()
masksImages = masks[:,7]
# saturImages = masks[:,8]

os.makedirs('Data/MapsStaple3class', exist_ok=True)

for imnum in range(len(histoImages)):
    # preprocess mask
    ma = io.imread(str(masksImages[imnum]))
    ma[ma <= 2] = 0
    ma[ma == 3] = 1
    ma[ma >= 4] = 2
    
    
    imageIndex = str(histoImages[imnum].replace('.jpg', '').replace('Data/Train Imgs/', ''))
    io.imsave(str('Data/MapsStaple3class/'+str(imageIndex)+'.png'), ma.astype(np.uint8), check_contrast= False)
    