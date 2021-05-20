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

from create_data import load_data

from skimage.measure import label, regionprops
from sklearn.metrics import cohen_kappa_score



def get_score(sop, thres = 13390*16):
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



# def get_score_unet(sop, thres = 13390, maxi = []):
#     ### per default Benign
#     score = 0
#     ## gleason 3+3
#     if sop[1]>thres and sop[2]<=thres:
#         score = 1
#     ## gleason 3+4
#     elif sop[1]>thres and sop[2]>thres and sop[1]>sop[2]:
#         score = 2
#         if maxi[2]<thres:
#             score = 1
#     ## gleason 4+3
#     elif sop[1]>thres and sop[2]>thres and sop[1]<=sop[2]:
#         score = 3
#         if maxi[1]<thres:
#             score = 4
#     ## gleason 4+4
#     elif sop[1]<=thres and sop[2]>thres:
#         score = 4
#     return score


def get_score_unet(sop, thres = 13390, maxi = []):
    ### per default Benign
    score = 0
    ## gleason 3+3
    if maxi[1]>thres and maxi[2]<=thres:
        score = 1
    ## gleason 3+4
    elif maxi[1]>thres and maxi[2]>thres and sop[1]>sop[2]:
        score = 2
    ## gleason 4+3
    elif maxi[1]>thres and maxi[2]>thres and sop[1]<=sop[2]:
        score = 3
    ## gleason 4+4
    elif maxi[1]<=thres and maxi[2]>thres:
        score = 4
    return score


def rk_score_conmat(imlist):
    ### metric on data set (expert and consensus)
    full_array = np.zeros((8,8,5,5))
    for i in range(8):
        for j in range(i, 8):
            for imnum in imlist:
                if masks[imnum,i] != 'NoGT' and masks[imnum,j] != 'NoGT':
                    truemap = io.imread(str(masks[imnum,i]))
                    predmap = io.imread(str(masks[imnum,j]))
                    truemap[truemap <= 2] = 0; truemap[truemap == 3] = 1; truemap[truemap >= 4] = 2
                    predmap[predmap <= 2] = 0; predmap[predmap == 3] = 1; predmap[predmap >= 4] = 2
                    true = np.array([np.count_nonzero(truemap==0), np.count_nonzero(truemap==1), np.count_nonzero(truemap==2)])
                    pred = np.array([np.count_nonzero(predmap==0), np.count_nonzero(predmap==1), np.count_nonzero(predmap==2)])
                    full_array[i, j, get_score(true), get_score(pred)] += 1
                    
    # os.makedirs('Scorestest/', exist_ok=True)
    # np.save('Scorestest/GleasonScore8x8.npy', full_array.astype(np.uint8))
    return full_array




def rk_score_conmat_unet(imlist, imdata, dirs_staexp, bias=13390):
     
    ### metric on unet
    full_array = np.zeros((8,1,5,5))
    for i in range(8):
        # offset=0
        for imnum in imlist:
            if masks[imnum,i] != 'NoGT':
                imageIndex = str(histoImages[imnum].replace('.jpg', '.png').replace('Data/Train Imgs/', '/'))
                truemap = io.imread(str(masks[imnum,i]))
                truemap[truemap <= 2] = 0; truemap[truemap == 3] = 1; truemap[truemap >= 4] = 2
                true = np.array([np.count_nonzero(truemap==0), np.count_nonzero(truemap==1), np.count_nonzero(truemap==2)])            
                
                # count = np.count_nonzero(imdata[:,0]==imnum)
                # data_array = imdata[offset : offset + count]
                # offset += count
    
                # i0 = np.min(data_array[:, 1])%reso
                # j0 = np.min(data_array[:, 2])%reso

                predmap = io.imread(str('Unets/' + dirs_staexp + imageIndex))
                i0 = np.argwhere(predmap >= 1)[0,0]%reso
                j0 = np.argwhere(predmap >= 1)[0,1]%reso
                predmap = predmap[i0::reso,j0::reso]               
                
                maxarea = [0,0,0]
                for j in range(2):    
                    pred = np.zeros((predmap.shape[:2]))             
                    pred = (predmap == j+1).astype(int)
                    lab = label(pred)            
                    objects = regionprops(lab)
                    if objects:
                        maxarea[j+1] = np.max([x.area for x in objects])
                    else:
                        maxarea[j+1] = 0            
                pred = np.array([np.count_nonzero(predmap==0), np.count_nonzero(predmap==1), np.count_nonzero(predmap==2)])

                full_array[i, 0, get_score(true, 13390*16),  get_score_unet(pred, bias, maxi=maxarea)] += 1
    return full_array



def get_rk(conmat_array):
    
    rk_array = np.zeros((conmat_array.shape[0:2]))
    for i in range(conmat_array.shape[0]):
        for j in range(conmat_array.shape[1]):
            
            conf = conmat_array[i,j]
            ### Rk statistic
            tk= np.sum(conf,axis=1)         ## number of times class k truly occurred
            pk= np.sum(conf,axis=0)         ## number of times class k was predicted
            c = np.sum([conf[i][i] for i in range(5)])    ## total number of samples correctly predicted
            s = np.sum(conf) ## total number of samples
            rknum = np.int(c*s) - np.sum(tk*pk)
            rkden = np.sqrt(np.int(s**2)-np.sum(pk*pk)) * np.sqrt(np.int(s**2)-np.sum(tk*tk)) + np.finfo(float).eps
            rk = rknum / rkden
            
            rk_array[i,j] = rk
    
    return rk_array

def bet_unet(imlist, imdata, bias=13390):
     
    ### metric on unet
    full_array = np.zeros((1,1,5,5))
    # for i in range(8):
    # offset=0
    for imnum in imlist:
        imageIndex = str(histoImages[imnum].replace('.jpg', '.png').replace('Data/Train Imgs/', '/'))
        
        # count = np.count_nonzero(imdata[:,0]==imnum)
        # data_array = imdata[offset : offset + count]
        # offset += count

        
        # i0 = np.min(data_array[:, 1])%reso
        # j0 = np.min(data_array[:, 2])%reso
        
        dirs2 = os.listdir('epochs/testing2/'+dirs[1])
        name = 'test2_20'

        dirs_staexp = name+'/'+dirs[1]+dirs2[16].replace('.h5', '')
        predmap = io.imread(str('Unets/' + dirs_staexp + imageIndex))
        i0 = np.argwhere(predmap >= 1)[0,0]%reso
        j0 = np.argwhere(predmap >= 1)[0,1]%reso
        predmap = predmap[i0::reso,j0::reso]
        
        maxarea = [0,0,0]
        for j in range(2):    
            pred = np.zeros((predmap.shape[:2]))             
            pred = (predmap == j+1).astype(int)
            lab = label(pred)            
            objects = regionprops(lab)
            if objects:
                maxarea[j+1] = np.max([x.area for x in objects])
            else:
                maxarea[j+1] = 0
        pred = np.array([np.count_nonzero(predmap==0), np.count_nonzero(predmap==1), np.count_nonzero(predmap==2)])
        score_staple = get_score_unet(pred, bias, maxi=maxarea)
        
        dirs2 = os.listdir('epochs/testing2/'+dirs[1])
        name = 'test2_00'

        dirs_staexp = name+'/'+dirs[1]+dirs2[16].replace('.h5', '')
        predmap = io.imread(str('Unets/' + dirs_staexp + imageIndex))[i0::reso,j0::reso]
        maxarea = [0,0,0]
        for j in range(2):    
            pred = np.zeros((predmap.shape[:2]))             
            pred = (predmap == j+1).astype(int)
            lab = label(pred)            
            objects = regionprops(lab)
            if objects:
                maxarea[j+1] = np.max([x.area for x in objects])
            else:
                maxarea[j+1] = 0
        pred = np.array([np.count_nonzero(predmap==0), np.count_nonzero(predmap==1), np.count_nonzero(predmap==2)])
        score_expert = get_score_unet(pred, bias, maxi=maxarea)
        full_array[0,0, score_expert, score_staple] += 1
        
    return full_array



def get_labels(imlist):
    ### metric on data set (expert and consensus)
    full_array = np.zeros((8,8))
    for i in range(8):
        for j in range(i, 8):
            lab_pred = []
            lab_true = []
            for imnum in imlist:
                if masks[imnum,i] != 'NoGT' and masks[imnum,j] != 'NoGT':
                    
                    truemap = io.imread(str(masks[imnum,i]))
                    predmap = io.imread(str(masks[imnum,j]))
                    truemap[truemap <= 2] = 0; truemap[truemap == 3] = 1; truemap[truemap >= 4] = 2
                    predmap[predmap <= 2] = 0; predmap[predmap == 3] = 1; predmap[predmap >= 4] = 2
                    true = np.array([np.count_nonzero(truemap==0), np.count_nonzero(truemap==1), np.count_nonzero(truemap==2)])
                    pred = np.array([np.count_nonzero(predmap==0), np.count_nonzero(predmap==1), np.count_nonzero(predmap==2)])
                    lab_true.append(get_score(true))
                    lab_pred.append(get_score(pred))
            
            full_array[i, j] = cohen_kappa_score(lab_true, lab_pred, labels=[0, 1, 2, 3, 4])
            print(full_array[i, j])
    # os.makedirs('Scorestest/', exist_ok=True)
    # np.save('Scorestest/GleasonScore8x8.npy', full_array.astype(np.uint8))
    return full_array



def test_kappa(imlist, imdata, dirs_staexp, bias=13390):
     
    ### metric on unet
    full_array = np.zeros((8,1))
    for i in range(8):
        lab_pred = []
        lab_true = []
        for imnum in imlist:
            if masks[imnum,i] != 'NoGT':
                imageIndex = str(histoImages[imnum].replace('.jpg', '.png').replace('Data/Train Imgs/', '/'))
                truemap = io.imread(str(masks[imnum,i]))
                truemap[truemap <= 2] = 0; truemap[truemap == 3] = 1; truemap[truemap >= 4] = 2
                true = np.array([np.count_nonzero(truemap==0), np.count_nonzero(truemap==1), np.count_nonzero(truemap==2)])            
                
                # count = np.count_nonzero(imdata[:,0]==imnum)
                # data_array = imdata[offset : offset + count]
                # offset += count
    
                # i0 = np.min(data_array[:, 1])%reso
                # j0 = np.min(data_array[:, 2])%reso

                predmap = io.imread(str('Unets/' + dirs_staexp + imageIndex))
                i0 = np.argwhere(predmap >= 1)[0,0]%reso
                j0 = np.argwhere(predmap >= 1)[0,1]%reso
                predmap = predmap[i0::reso,j0::reso]               
                
                maxarea = [0,0,0]
                for j in range(2):    
                    pred = np.zeros((predmap.shape[:2]))             
                    pred = (predmap == j+1).astype(int)
                    lab = label(pred)            
                    objects = regionprops(lab)
                    if objects:
                        maxarea[j+1] = np.max([x.area for x in objects])
                    else:
                        maxarea[j+1] = 0            
                pred = np.array([np.count_nonzero(predmap==0), np.count_nonzero(predmap==1), np.count_nonzero(predmap==2)])
    
                lab_true.append(get_score(true, 13390*16))
                lab_pred.append(get_score_unet(pred, bias, maxi=maxarea))
            
        full_array[i, 0] = cohen_kappa_score(lab_true, lab_pred, labels=[0, 1, 2, 3, 4])
            
    return full_array



if __name__ == '__main__':

    histoImages, masks = load_data()
    
    # network
    patchsize = 128
    reso = 4
    features = 48
    blocks = 5
    bias = 13390

    # loading
    [mean, std] = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/meanstd.npy')    
    val = np.load('Processed/ps' + str(patchsize) + 'reso' + str(reso)+'select/testdata.npy')
    val = val[val[:, 0].argsort()]
    val[:,0] = val[:,0]+167
    imnumval = np.unique(val[:,0])
    # imnumval = np.arange(167)
        
    # model = UNet((patchsize,patchsize,3), filters=features, blocks=blocks) 

    # arr = rk_score_conmat(imnumval)
    # rk = get_rk(arr)
    # print(rk)
    """
    name = 'test2_00'

    dirs = os.listdir('epochs/testing2')
    
    dirs2 = os.listdir('epochs/testing2/'+dirs[0])
    dirs_sta = name+'/'+dirs[0]+dirs2[13].replace('.h5', '')
    dirs2 = os.listdir('epochs/testing2/'+dirs[1])
    dirs_exp = name+'/'+dirs[1]+dirs2[16].replace('.h5', '')    
    
    sta_array = test_kappa(imnumval, val, dirs_sta, bias=bias)
    
    exp_array = test_kappa(imnumval, val, dirs_exp, bias=bias)
    
    print('sta_array ', sta_array )
    print('exp_array ', exp_array )
    """
    
    
    """
    #############################
    name = 'test2_00'
    dirs = os.listdir('epochs/testing2')
    
    arr = bet_unet(imnumval, val, bias=bias)
    rk = get_rk(arr)
    print('arr', arr[0,0])
    print('bet_unet', rk)
    
    
    name = 'val2_00'

    dirs = os.listdir('epochs/validation2')
    dirs2 = os.listdir('epochs/validation2/'+dirs[0])
    dirs_sta = name+'/'+dirs[0]+dirs2[13].replace('.h5', '')
    dirs2 = os.listdir('epochs/validation2/'+dirs[1])
    dirs_exp = name+'/'+dirs[1]+dirs2[16].replace('.h5', '')    
    
    os.makedirs('Scoresval/', exist_ok=True)
    sta_array = rk_score_conmat_unet(imnumval, val, dirs_sta, bias=bias)
    np.save('Scoresval/GleasonScore8x1_staple.npy', sta_array.astype(np.uint8))
    
    exp_array = rk_score_conmat_unet(imnumval, val, dirs_exp, bias=bias)
    np.save('Scoresval/GleasonScore8x1_expert.npy', exp_array.astype(np.uint8))
    
    sta_rk = get_rk(sta_array)
    exp_rk = get_rk(exp_array)
    print('sta_rk ', sta_rk )
    print('exp_rk ', exp_rk )
    
    #############################
    # full_array = rk_score_conmat(imnumval)
    # rks = get_rk(full_array)
    # print('rks \n', rks )
    
    
    
    #### create 8x8 rk score
    conmat = rk_score_conmat(imnumtrain, name="train")
    rk_t = get_rk(conmat, "train")
    rk_t = rk_t
    
    cmap = 'coolwarm'    
    row_labels = ["Expert 1", "Expert 2",  "Expert 3",  "Expert 4",
                  "Expert 5",  "Expert 6",  "Majority vote", "Staple"]
    col_labels = ["Expert 1", "Expert 2",  "Expert 3",  "Expert 4",
                  "Expert 5",  "Expert 6",  "Majority vote", "Staple"]
    
    rk_t = np.round(rk_t,2)
    df = pd.DataFrame(rk_t,row_labels,col_labels)

    plt.figure()
    plt.figure(figsize = [13.5,10.5])
    
    sns.heatmap(df, cmap = cmap, annot=True, fmt='g')
    
    # title = 'Influence of threshold on softmax and offset/% to consider a grade'
    # plt.title(title, fontsize = 18) # title with fontsize 20
    plt.xlabel('Predicted', fontsize = 15) # x-axis label with fontsize 15
    plt.ylabel('Actual', fontsize = 15) # y-axis label with fontsize 15
    plt.yticks(rotation=0) 
    
    os.makedirs("Scorestrain", exist_ok=True)
    plt.savefig("Scorestrain/rkscoretrain.png")


    #### create 8x8 rk score
    conmat = rk_score_conmat(imnumval, name="val")
    rk = get_rk(conmat, "val")
    
    
    cmap = 'coolwarm'    
    row_labels = ["Expert 1", "Expert 2",  "Expert 3",  "Expert 4",
                  "Expert 5",  "Expert 6",  "Majority vote", "Staple"]
    col_labels = ["Expert 1", "Expert 2",  "Expert 3",  "Expert 4",
                  "Expert 5",  "Expert 6",  "Majority vote", "Staple"]
    
    rk = np.round(rk,2)
    df = pd.DataFrame(rk,row_labels,col_labels)

    plt.figure()
    plt.figure(figsize = [13.5,10.5])
    
    sns.heatmap(df, cmap = cmap, annot=True, fmt='g')
    
    # title = 'Influence of threshold on softmax and offset/% to consider a grade'
    # plt.title(title, fontsize = 18) # title with fontsize 20
    plt.xlabel('Predicted', fontsize = 15) # x-axis label with fontsize 15
    plt.ylabel('Actual', fontsize = 15) # y-axis label with fontsize 15
    plt.yticks(rotation=0) 
    
    os.makedirs("Scorestrain", exist_ok=True)
    plt.savefig("Scorestrain/rkscoreval.png")

"""
    
    