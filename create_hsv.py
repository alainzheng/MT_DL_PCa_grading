
# -*- coding: utf-8 -*-
"""

get saturation mask with threshold ans puts it into folder -> saturTh = 0.15
in folder: /Data/Train hsv_sat/

Created on Thu Nov 26 10:55:25 2020

@author: Alain
"""

import numpy as np
from skimage import io
from create_data import load_data
import time
import datetime
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2hsv



tic = time.perf_counter()


print('-'*30)
print('running saturation.py ...')
print('-'*30)

saturThres = 0.15 # saturation limit below which we consider it is background
PATCHSIZE = 256 ##  try 256, 512


histoImages, masks = load_data()
os.makedirs(str('Data/Train hsv_sat'), exist_ok=True)

for imnum in range(len(histoImages)):
    im = io.imread(str(histoImages[imnum]))
    
    saturIm = rgb2hsv(im)[:,:,1]
    
    saturIm[saturIm<saturThres] = 0
    saturIm[saturIm>=saturThres] = 1

    # hsv2 = np.zeros((im.shape[:2]))
    # for i in range(0,hsv.shape[0],PATCHSIZE):
    #     for j in range(0,hsv.shape[1],PATCHSIZE):
    #         patch = hsv[i:i+PATCHSIZE,j:j+PATCHSIZE]
    #         if np.count_nonzero(patch==1) >= PATCHSIZE**2 * 0.70:
    #             hsv2[i:i+PATCHSIZE,j:j+PATCHSIZE] = 1

    imageIndex = str(histoImages[imnum].replace('.jpg', '').replace('Data/Train Imgs/', ''))
    io.imsave(str('Data/Train hsv_sat/'+str(imageIndex)+'.png'), saturIm.astype(np.uint8), check_contrast= False)
    


toc = time.perf_counter()
print('toc-tic: ' + str(datetime.timedelta(seconds = toc-tic)))
  
