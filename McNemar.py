# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:43:34 2021



get rk per classification -> not pixel

@author: Alain
"""



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

def get_score(sop, thres = 13390, mini = 0.05):
    ### per default Benign
    score = 0.1
    ## gleason 3+3
    if sop[1]>thres and sop[2]<=thres:
        score = 1
    ## gleason 3+4
    elif sop[1]>thres and sop[2]>thres and sop[1]>sop[2]:
        score = 2
    ## gleason 4+3
    elif sop[1]>thres and sop[2]>thres and sop[1]<=sop[2]:
        if sop[1]>mini*(sop[2]+sop[1]):
            score = 3
        else:
            score = 4
    ## gleason 4+4
    elif sop[1]<=thres and sop[2]>thres:
        score = 4
    return score

def get_score_unet(sop, thres = 13300, maxi = []):
    ### per default Benign
    mini= 0.1
    score = 0
    ## gleason 3+3
    if sop[1]>thres and sop[2]<=thres:
        score = 1
    ## gleason 3+4
    elif sop[1]>thres and sop[2]>thres and sop[1]>sop[2]:
        if sop[2]>mini*(sop[2]+sop[1]):
            score = 2
        else:
            score = 1
    ## gleason 4+3
    elif sop[1]>thres and sop[2]>thres and sop[1]<=sop[2]:
        if sop[1]>mini*(sop[2]+sop[1]):
            score = 3
        else:
            score = 4
    ## gleason 4+4
    elif sop[1]<=thres and sop[2]>thres:
        score = 4
    return score

"""
def get_score(sop, thres = 13390):
    ### per default Benign
    score = 0
    ## gleason 3+3
    if sop[1]>thres and sop[2]<=thres:
        score = 1
    ## gleason 3+4
    elif sop[1]>thres and sop[2]>thres and sop[1]>sop[2]:
        score = 2
    ## gleason 4+3
    elif sop[1]>thres and sop[2]>thres and sop[1]<=sop[2]:
        score = 3
    ## gleason 4+4
    elif sop[1]<=thres and sop[2]>thres:
        score = 4
    return score

def get_score_unet(sop, thres = 13300, maxi = []):
    ### per default Benign
    score = 0
    ## gleason 3+3
    if sop[1]>thres and sop[2]<=thres:
        score = 1
    ## gleason 3+4
    elif sop[1]>thres and sop[2]>thres and sop[1]>sop[2]:
        score = 2
        if maxi[2]<thres:
            score = 1
    ## gleason 4+3
    elif sop[1]>thres and sop[2]>thres and sop[1]<=sop[2]:
        score = 3
        if maxi[1]<thres:
            score = 4
    ## gleason 4+4
    elif sop[1]<=thres and sop[2]>thres:
        score = 4
    return score
"""


def get_binary_staple(imlist, imdata, dirs_staexp, sup=7, bias=0):
    
    ### metric on unet
    full_array = np.zeros((len(imlist)))
    # conmat = np.zeros((5,5))

    k=0
    # offset=0
    a = []
    for imnum in imlist:
        if masks[imnum, sup] != 'NoGT':
            
            # count = np.count_nonzero(imdata[:,0]==imnum)
            # data_array = imdata[offset : offset + count]
            # offset += count

            # i0 = np.min(data_array[:, 1])%reso
            # j0 = np.min(data_array[:, 2])%reso
            
            imageIndex = str(histoImages[imnum].replace('.jpg', '.png').replace('Data/Train Imgs/', '/'))

            truemap = io.imread(str(masks[imnum,sup]))
            truemap[truemap <= 2] = 0; truemap[truemap == 3] = 1; truemap[truemap >= 4] = 2
            true = np.array([np.count_nonzero(truemap==0), np.count_nonzero(truemap==1), np.count_nonzero(truemap==2)])
            
            predmap = io.imread(str('Unets/' + dirs_staexp + imageIndex))
            i0 = np.argwhere(predmap >= 1)[0,0]%reso
            j0 = np.argwhere(predmap >= 1)[0,1]%reso
            predmap = predmap[i0::reso,j0::reso]
            
            maxarea = [0,0,0]
            for i in range(2):
                
                pred = np.zeros((predmap.shape[:2]))             
                pred = (predmap == i+1).astype(int)
                lab = label(pred)            
                objects = regionprops(lab)
                if objects:
                    maxarea[i+1] = np.max([x.area for x in objects])
                else:
                    maxarea[i+1] = 0            
            pred = np.array([np.count_nonzero(predmap==0), np.count_nonzero(predmap==1), np.count_nonzero(predmap==2)])
            if  get_score(true, 13390*16) == get_score_unet(pred, bias, maxi=maxarea):
                full_array[k] = 1
            k+=1
            # conmat[get_score(true, 13390*16), get_score_unet(pred, bias, maxi=maxarea)] +=1
            # print(dirs_staexp, '\n', conmat)
            # tissue = np.count_nonzero(val[:,0]==imnum)*128*128
            # print("undecided %: ", np.count_nonzero(predmap==3)/tissue)
            # a.append(np.count_nonzero(predmap==3)/tissue)
    return full_array, a







if __name__ == '__main__':

    
    tic = time.perf_counter()

    histoImages, masks = load_data()
    
    # network
    patchsize = 128
    reso = 4
    features = 48
    blocks = 5
    bias = 13390

    # loading
    [mean, std] = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/meanstd.npy')    
    train = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/traindata.npy')
    val = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/testdata.npy')
    train = train[train[:, 0].argsort()]    
    val = val[val[:, 0].argsort()]
    val[:,0] = val[:,0]+167
    imnumtrain = np.unique(train[:,0])
    imnumval = np.unique(val[:,0])
        
    # model = UNet((patchsize,patchsize,3), filters=features, blocks=blocks) 

    ############################# 0   1 
    ############################# 13  16
    
    name = "test2_00"
    expn = 16
    stan= 13
    dirs = os.listdir('epochs/testing2')

    
    #############################
    ##   test2_00
    #############################
    dirs2 = os.listdir('epochs/testing2/'+dirs[0])
    dirs_sta = name+'/'+dirs[0]+dirs2[stan].replace('.h5', '')
    dirs2 = os.listdir('epochs/testing2/'+dirs[1])
    dirs_exp = name+'/'+dirs[1]+dirs2[expn].replace('.h5', '')

    sta_array, a = get_binary_staple(imnumval, val, dirs_sta, sup=7, bias=bias)  ### also return an array
    exp_array, b = get_binary_staple(imnumval, val, dirs_exp, sup=7, bias=bias)  ### also return an array

    # print('staple: \n', sta_array)
    # print('expert: \n', exp_array)
    
    conf_mat = confusion_matrix(sta_array, exp_array) 
    print('conmat: \n', conf_mat)
    
    result = mcnemar(conf_mat, exact=True)
    print('mcnemar test, binomial distribution  !!  test2_00')
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    
    
    #############################
    ##   test2_20
    #############################
    name = 'test2_20'
    dirs2 = os.listdir('epochs/testing2/'+dirs[0])
    dirs_sta = name+'/'+dirs[0]+dirs2[stan].replace('.h5', '')
    dirs2 = os.listdir('epochs/testing2/'+dirs[1])
    dirs_exp = name+'/'+dirs[1]+dirs2[expn].replace('.h5', '')

    sta_array, a = get_binary_staple(imnumval, val, dirs_sta, sup=7, bias=bias)  ### also return an array
    exp_array, b = get_binary_staple(imnumval, val, dirs_exp, sup=7, bias=bias)  ### also return an array
    
    plt.figure(0)
    plt.title('%of pixel predictions remaining after threshold on softmax')
    plt.boxplot([1-np.array(a), 1-np.array(b)])
    plt.xticks([1, 2], ['STAPLE', 'expert'])

    # print('staple: \n', sta_array)
    # print('expert: \n', exp_array)
    
    conf_mat = confusion_matrix(sta_array, exp_array) 
    print('conmat: \n', conf_mat)
    
    result = mcnemar(conf_mat, exact=True)
    print('mcnemar test, binomial distribution  !!  test2_20')
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    
    
    ##########################
    ## expert
    ##########################

    name = "test2_00"
    dirs_sta = name+'/'+dirs[1]+dirs2[expn].replace('.h5', '')
    name = "test2_20"    
    dirs_exp = name+'/'+dirs[1]+dirs2[expn].replace('.h5', '')

    sta_array, a = get_binary_staple(imnumval, val, dirs_sta, sup=7, bias=bias)  ### also return an array
    exp_array, b = get_binary_staple(imnumval, val, dirs_exp, sup=7, bias=bias)  ### also return an array
    
    # print('test2_00: \n', sta_array)
    # print('test2_20: \n', exp_array)
    
    conf_mat = confusion_matrix(sta_array, exp_array) 
    print('conmat: \n', conf_mat)
    
    result = mcnemar(conf_mat, exact=True)
    print('mcnemar test, binomial distribution  !!  expert')
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    
    
    #######################
    ### staple
    #######################
    
    name = "test2_00"
    dirs_sta = name+'/'+dirs[0]+dirs2[stan].replace('.h5', '')
    name = "test2_20"    
    dirs_exp = name+'/'+dirs[0]+dirs2[stan].replace('.h5', '')

    sta_array, a = get_binary_staple(imnumval, val, dirs_sta, sup=7, bias=bias)  ### also return an array
    exp_array, b = get_binary_staple(imnumval, val, dirs_exp, sup=7, bias=bias)  ### also return an array
    
    # print('test2_00: \n', sta_array)
    # print('test2_20: \n', exp_array)

    conf_mat = confusion_matrix(sta_array, exp_array) 
    print('conmat: \n', conf_mat)
    
    result = mcnemar(conf_mat, exact=True)
    print('mcnemar test, binomial distribution  !! staple')
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    
    
    ########################################
    #############    TIMER     #############
    ########################################
    
    toc = time.perf_counter()
    print('toc-tic: ' + str(datetime.timedelta(seconds = toc-tic)))
      





"""
if __name__ == '__main__':

    
    tic = time.perf_counter()

    histoImages, masks = load_data()
    
    ########################################
    #############    TIMER     #############
    ########################################
    counter = np.array([])
    for exp in range(6):
        for imnum in range(len(histoImages)):
            if masks[imnum, exp] != 'NoGT':
                imageIndex = str(histoImages[imnum].replace('.jpg', '.png').replace('Data/Train Imgs/', '/'))
    
                truemap = io.imread(str(masks[imnum,exp]))
                # truemap[truemap <= 2] = 0; truemap[truemap == 3] = 1; truemap[truemap >= 4] = 2
                true = np.array([np.count_nonzero(truemap==1), np.count_nonzero(truemap==2), 
                                 np.count_nonzero(truemap==3), np.count_nonzero(truemap==4),
                                 np.count_nonzero(truemap==5)])
                
                counter = np.append(counter, true[true!=0])
                for i in range(len(true)):
                    if true[i]<250000 and true[i]!=0:
                        print('exp: ', exp+1, 'image: ', imageIndex)
                        print('count: ', true[i], ' -- of grade:', i+1)
                
    counter = counter[counter.argsort()]    

    print(counter[:20])
    toc = time.perf_counter()
    print('toc-tic: ' + str(datetime.timedelta(seconds = toc-tic)))
      

"""