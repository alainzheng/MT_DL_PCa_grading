# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:28:26 2021

@author: Alain
"""


import tensorflow as tf
import numpy as np
import random

## for gaussian noise
mean = 0
var = 0.1
sigma = var**0.5
random.seed(16)

def get_example(image, labels, prob=0.8):
    
    augment = random.random()
    if augment < prob:
        return get_augmented_data(image, labels)
    else:
        return image, labels


def get_augmented_data(image, labels):
    
    augmentation_index = random.randint(1,4)
    
    if augmentation_index == 1:
        return np.fliplr(image), np.fliplr(labels)
    elif augmentation_index == 2:
        return np.flipud(image), np.flipud(labels)
    elif augmentation_index == 3:
        return np.rot90(image), np.rot90(labels)
    else: 
        gauss = np.random.normal(0,sigma,(image.shape))
        noisy_image = image + gauss
        return noisy_image, labels

