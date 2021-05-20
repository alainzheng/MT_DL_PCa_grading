# -*- coding: utf-8 -*-
"""
create array of histologic images and array of 6 maps + 2 consensus maps 

second code to run
But first run the consensus.py to create the 2 additional data files 
(majority vote and STAPLE)

@author: alain zheng

"""


import os
import numpy as np
import time
import datetime


def create_images(path):
    slices = [(path + '/' + s) for s in os.listdir(path)]
    return slices


def create_data():
    
    dataDirList = os.listdir('Data/')
    histoImages = create_images('Data/' + dataDirList[-1]) # create list of HE images
    
    groundTruths = []  # create a set of list of maps, groundTruths[dataDirList][imageNumber]
    for h in range(len(dataDirList)-1):
        groundTruths.append(create_images('Data/' + dataDirList[h]))
        
    masks = []
    for singleHistoImage in histoImages: #for every image in Train Imgs
        mask = (len(dataDirList)-1)*['NoGT']
        sliceName = singleHistoImage.replace('.jpg', '').replace('Data/Train Imgs/', '') #ex: slide006_core077
        for expertNumber in range(len(dataDirList)-1): # for every expert
            for gtNumber in range(len(groundTruths[expertNumber])):
                if sliceName in groundTruths[expertNumber][gtNumber]:
                    mask[expertNumber] = groundTruths[expertNumber][gtNumber]
        masks.append(mask)
        
    
    np.save('Datanpy/images.npy', histoImages)
    np.save('Datanpy/masks.npy', masks)
    
    
    
    
def create_train_data():
    
    dataDirList = os.listdir('Data/')
    histoImages = create_images('Data/' + dataDirList[-1])[:167] # create list of HE images
    
    groundTruths = []  # create a set of list of maps, groundTruths[dataDirList][imageNumber]
    for h in range(len(dataDirList)-1):
        groundTruths.append(create_images('Data/' + dataDirList[h]))
        
    masks = []
    for singleHistoImage in histoImages: #for every image in Train Imgs
        mask = (len(dataDirList)-1)*['NoGT']
        sliceName = singleHistoImage.replace('.jpg', '').replace('Data/Train Imgs/', '') #ex: slide006_core077
        for expertNumber in range(len(dataDirList)-1): # for every expert
            for gtNumber in range(len(groundTruths[expertNumber])):
                if sliceName in groundTruths[expertNumber][gtNumber]:
                    mask[expertNumber] = groundTruths[expertNumber][gtNumber]
        masks.append(mask)
    
    np.save('Datanpy/trainImages.npy', histoImages)
    np.save('Datanpy/trainMasks.npy', masks)
    
    
def create_test_data():
    
    dataDirList = os.listdir('Data/')
    histoImages = create_images('Data/' + dataDirList[-1])[167:] # create list of HE images
    
    groundTruths = []  # create a set of list of maps, groundTruths[dataDirList][imageNumber]
    for h in range(len(dataDirList)-1):
        groundTruths.append(create_images('Data/' + dataDirList[h]))
        
    masks = []
    for singleHistoImage in histoImages: #for every image in Train Imgs
        mask = (len(dataDirList)-1)*['NoGT']
        sliceName = singleHistoImage.replace('.jpg', '').replace('Data/Train Imgs/', '') #ex: slide006_core077
        for expertNumber in range(len(dataDirList)-1): # for every expert
            for gtNumber in range(len(groundTruths[expertNumber])):
                if sliceName in groundTruths[expertNumber][gtNumber]:
                    mask[expertNumber] = groundTruths[expertNumber][gtNumber]
        masks.append(mask)
    
    np.save('Datanpy/testImages.npy', histoImages)
    np.save('Datanpy/testMasks.npy', masks)
    


def load_train_data():

    imgs = np.load('Datanpy/trainImages.npy')

    masks = np.load('Datanpy/trainMasks.npy')

    return imgs, masks

def load_test_data():

    imgs = np.load('Datanpy/testImages.npy')

    masks = np.load('Datanpy/testMasks.npy')

    return imgs, masks


def load_data():

    imgs = np.load('Datanpy/images.npy')

    masks = np.load('Datanpy/masks.npy')

    return imgs, masks


if __name__ == '__main__':
    
    print('-'*30)

    print('Data creation...')

    print('-'*30)
    
    tic = time.perf_counter()


    create_train_data()
    
    create_test_data()
    
    create_data()
    
    x,y = load_train_data()
    
    toc = time.perf_counter()
    print('toc-tic: ' + str(datetime.timedelta(seconds = toc-tic)))
    
    
    
    