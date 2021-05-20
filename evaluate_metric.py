# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:43:34 2021

@author: Alain
"""



import numpy as np
from skimage import io
import time
import datetime
import matplotlib.pyplot as plt
import os
from unet import UNet
from tensorflow.keras.utils import to_categorical

from create_data import load_data



def rk_stat_conmat(imdata, imlist, name="val"):

    
    ### metric on validation set (expert and consensus)
    rk_array = np.zeros((8,8))
    size_array = np.zeros((8,8))
    mcc = np.zeros((8, 8, 3))
    for i in range(8):
        for j in range(i, 8):
            ### for every combination of true and pred
            TP = np.zeros(3); TN = np.zeros(3); FP = np.zeros(3); FN = np.zeros(3)
            met_size = 0 
            
            for imnum in imlist:
                if masks[imnum,i] != 'NoGT' and masks[imnum,j] != 'NoGT':
                    met_size+=1
                    
                    truemap = io.imread(str(masks[imnum,i]))
                    predmap = io.imread(str(masks[imnum,j]))
                    truemap[truemap <= 2] = 0; truemap[truemap == 3] = 1; truemap[truemap >= 4] = 2
                    predmap[predmap <= 2] = 0; predmap[predmap == 3] = 1; predmap[predmap >= 4] = 2
                    truemap = to_categorical(truemap, num_classes=3)
                    predmap = to_categorical(predmap, num_classes=3)
                    
                    data_array = imdata[imdata[:,0]==imnum]
                    # print(np.unique(data_array[:,0]))
                    
                    for patch in range(len(data_array)):
                        true = truemap[data_array[patch, 1]:data_array[patch, 1]+patchsize*reso:reso,
                                       data_array[patch, 2]:data_array[patch, 2]+patchsize*reso:reso]
                        pred = predmap[data_array[patch, 1]:data_array[patch, 1]+patchsize*reso:reso,
                                       data_array[patch, 2]:data_array[patch, 2]+patchsize*reso:reso]

                        ### every single returns a [class 1, class 2, class 3]
                        singleTP = np.sum( true * pred, axis=(0,1) )            # = TP
                        singleTN = np.sum( (1-true) * (1-pred), axis=(0,1) )    # = TN
                        
                        singleFP = np.sum( (1-true) * pred, axis=(0,1) )        # = FP            
                        singleFN = np.sum( true * (1-pred), axis=(0,1) )        # = FN
                        
                        
                        ### the following are lists and not numpy arrays
                        TP = TP + singleTP
                        TN = TN + singleTN
                        FP = FP + singleFP
                        FN = FN + singleFN
                
            
            ### mcc per class
            numerator = (TP * TN - FP * FN)
            denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + np.finfo(float).eps
            mcc[i,j] = numerator / denominator        
                   
            ### Rk statistic
            tk= TP+FN         ## number of times class k truly occurred
            pk= TP+FP         ## number of times class k was predicted
            c = np.sum(TP)    ## total number of samples correctly predicted
            s = np.sum(TP+FN) ## total number of samples
            rknum = int(c*s)-np.sum(tk*pk)
            rkden = np.sqrt(int(s**2)-np.sum(pk*pk)) * np.sqrt(int(s**2)-np.sum(tk*tk)) + np.finfo(float).eps
            rk = rknum / rkden
            
            rk_array[i,j] = rk
            size_array[i,j] = met_size
            
    os.makedirs('Scores2/rk_stat', exist_ok=True)
    np.save('Scores2/rk_stat/sd1GleasonGrade8x8_'+name+'.npy', rk_array)
    np.save('Scores2/rk_stat/sd1mcc.npy', mcc)
    np.save('Scores2/sd1effectif'+name+'.npy', size_array)
    return rk_array, mcc, size_array




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
    val = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/testdata.npy')
    val = val[val[:, 0].argsort()]
    val[:,0] = val[:,0]+167

    imnumval = np.unique(val[:,0])
    
        
    # model = UNet((patchsize,patchsize,3), filters=features, blocks=blocks) 

    

    #############################
    # train_num = 0    ### used to choose the training 
    # epoch_num = 1    ### used to choose the weight in the training (epoch)
    # dirs = os.listdir('epochs/validation2')
    
    
    a,b,c = rk_stat_conmat(val, imnumval, name="test")  
    
    
    ########################################
    #############    TIMER     #############
    ########################################
    
    toc = time.perf_counter()
    print('toc-tic: ' + str(datetime.timedelta(seconds = toc-tic)))
    
    
    
    """
    #############################
    for train_num in range(len(dirs)):
        dirs2 = os.listdir('epochs/validation2/'+dirs[train_num])

        for epoch_num in range(1,len(dirs2)):
            # loading weights
            # weights = 'epochs/validation2/'+dirs[train_num] +'/'+ dirs2[epoch_num]
            # model.load_weights(weights)
        

            #### create prediction
            # prediction(val, imnumval)
           
            
            #### create 1x8 rk stat wrt unet loop
            dirs_staexp = dirs[train_num] +'/' + dirs2[epoch_num].replace('.h5','')
            # rk_stat_conmat_unet(val, imnumval, dirs_staexp, name="val")
    
    
    #############################
    ###    compute all rk
    for train_num in range(len(dirs)):
        dirs2 = os.listdir('epochs/validation2/'+dirs[train_num])

        rks = np.zeros((8, len(dirs2)-1))
        for epoch_num in range(1,len(dirs2)):
            name = 'val'
            dirs_staexp = dirs[train_num] +'/' + dirs2[epoch_num].replace('.h5','')

            conmat_array = np.load('Scores2/rk_stat/'+ dirs_staexp +'/GleasonScore1x8_'+name+'.npy')
            rks[:, epoch_num-1] = conmat_array[:,0]
    
    
        os.makedirs('Scores2/rk_stat/'+ dirs[train_num], exist_ok=True)
        np.save('Scores2/rk_stat/'+ dirs[train_num] +'/rk_8xN_'+name+'.npy', rks)
        
    """
    ####################
    
    # rks = np.load('Scores2/rk_stat/'+ dirs[0] +'/rk_8xN_'+name+'.npy')
    # rks2 = np.load('Scores2/rk_stat/'+ dirs[1] +'/rk_8xN_'+name+'.npy')

    
    # x = np.arange(0,len(rks2[0]))
    # plt.figure(figsize = [12, 10])
    # for i in range(8):
    #     plt.plot(x,rks2[i])
    
    # plt.legend(["Expert 1", "Expert 2",  "Expert 3",  "Expert 4",
    #              "Expert 5",  "Expert 6",  "Majority vote", "Staple"])
    # os.makedirs("Figures", exist_ok=True)
    # plt.savefig("Figures/rkstat_exp.png")
    

    #### create 8x8 rk stat
    # rk_stat_conmat(val, imnumval, name="val")  
    
    #### create 1x1 rk stat wrt unet
    # dirs_staple = dirs[train_num] +'/' + dirs2[epoch_num].replace('.h5','')
    # dirs_expert = dirs[train_num] +'/' + dirs2[epoch_num].replace('.h5','')
    # rk_stat_conmat_inter_unets(val, imnumval, dirs_staple, dirs_expert, name="val")

    
    