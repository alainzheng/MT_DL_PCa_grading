"""

first code to run, makes the 2 consensus dataset (majority vote, staple)

"""

import numpy as np
import os
import matplotlib.pyplot as plt
import os.path as osp
from skimage import io
import SimpleITK as sitk
from dataPreproces import load_train_data,load_test_data,load_data
import time
import datetime
from collections import Counter


def find_majority(votes):
    
    ### handles ties in the sitk module
    vote_count = Counter(votes)
    top_count = vote_count.most_common()
    
    ### makes a list of most common elements and returns the maximum of them
    for i in range(len(top_count)-1,0,-1):
        if top_count[0][1]==top_count[i][1]:
            top_count = top_count[:i+1]
            break
    return max([top_count[j][0] for j in range(len(top_count))])    


def show_figure(item,outputdir):
    
    ##### make the figure
    im = sitk.ReadImage(osp.join(outputdir,item))
    plt.imshow(sitk.GetArrayViewFromImage(im), vmin=0, vmax=6) #les limites sont 0=> benign, 5 grade 5
    plt.colorbar()
            

def majority_second():
    
    #create numpy array corresponding to Majority vote GT
    for imageNumber in range(len(histoImages)):
            
        imageName = histoImages[imageNumber]
        imageIndex = imageName.replace('.jpg', '').replace('Data/Train Imgs/', '')
        truthMaps = masks[imageNumber,:6]
        ima = io.imread(imageName)
    
        mapstack = np.empty((ima.shape[0],ima.shape[1])) # to remove afterwards
        for i in range(len(truthMaps)):
            if truthMaps[i] != 'NoGT':
                im = io.imread(truthMaps[i])
                mapstack = np.dstack((mapstack,im))
        ## here mapstack will normally be of shape (height, width, number of annotated label + 1 )
        mapstack = mapstack[:,:,1:] # remove first layer
        
        map_mv = np.empty((ima.shape[0],ima.shape[1])) #consensus vote GT
        for j in range(ima.shape[0]):
            for k in range(ima.shape[1]):
                map_mv[j,k] = find_majority(mapstack[j,k,:])
                
        if not os.path.exists('Data/MapsMajority/'):
            os.makedirs('Data/MapsMajority/')   
        
        ### before saving, cast as np.uint8 to avoid conversion to uint8 afterward (no scaling therefore)
        io.imsave(str('Data/MapsMajority/'+str(imageIndex)+'.png'), map_mv.astype(np.uint8), check_contrast= False)
            
    

def majority(item):
    
    #### item format = 'slide001_core003_classimg_nonconvex.png'
    inputdirs = os.getcwd()+'/Data'
    outputdir = os.getcwd()+'/Data/MapsMajority/'
    os.makedirs(outputdir, exist_ok=True)
    
    imgs = [] #for an item, generates all the annotated maps (filenames)
    
    for p in os.listdir(inputdirs):
        if osp.isfile(osp.join('Data/'+p+'/', item)):
            imgs.append(sitk.ReadImage(osp.join('Data/'+p+'/', item)))
    

    result = sitk.LabelVoting(imgs, 255)
    # p1_max = np.max(sitk.GetArrayFromImage(imgs[0]))
    result_data = sitk.GetArrayFromImage(result)
    
    # result_data[result_data == 255] = p1_max
    ties = np.argwhere(result_data == 255)
    for tie in ties:
        result_data[tie[0],tie[1]]==find_majority([sitk.GetArrayFromImage(lb)[tie[0],tie[1]] for lb in imgs])
    
    result = sitk.GetImageFromArray(result_data)
    result.CopyInformation(imgs[0])
    sitk.WriteImage(result, osp.join(outputdir, item.replace('_classimg_nonconvex','')))


def staple(item):

    #### item = 'slide001_core007_classimg_nonconvex.png'
    inputdirs = os.getcwd()+'/Data'
    outputdir = os.getcwd()+'/Data/MapsStaple/'
    os.makedirs(outputdir, exist_ok=True)

    imgs = []

    for p in os.listdir(inputdirs):
        if osp.isfile(osp.join('Data/'+p+'/', item)):
            imgs.append(sitk.ReadImage(osp.join('Data/'+p+'/', item)))
            
    result = sitk.MultiLabelSTAPLE(imgs, 255)
    # p1_max = np.max(sitk.GetArrayFromImage(imgs[0]))
    result_data = sitk.GetArrayFromImage(result)
    
    # result_data[result_data == 255] = p1_max
    ties = np.argwhere(result_data == 255)
    for tie in ties:
        result_data[tie[0],tie[1]]==find_majority([sitk.GetArrayFromImage(lb)[tie[0],tie[1]] for lb in imgs])
    
    result = sitk.GetImageFromArray(result_data)
    result.CopyInformation(imgs[0])
    sitk.WriteImage(result, osp.join(outputdir, item.replace('_classimg_nonconvex','')))


if __name__ == '__main__':
    
    print('-'*30)

    print('Run consensus.py...')

    print('-'*30)
    
    tic = time.perf_counter()
    
    histoImages, masks = load_data()
    
    # outputdir = 'Data/MapsMajority/'  ## for show_figure argument
    # outputdir = 'Data/MapsStaple/'    ## for show_figure argument

    # imageIndex = 'slide001_core007_classimg_nonconvex.png' ## this format
    
    
    for imageName in histoImages:
        
        imageIndex = imageName.replace('.jpg', '_classimg_nonconvex.png').replace('Data/Train Imgs/', '')
        # staple(imageIndex)
        # majority(imageIndex)
        
    # show_figure(imageIndex,outputdir)
    
    
    toc = time.perf_counter()
    print('toc-tic: ' + str(datetime.timedelta(seconds = toc-tic)))
    
    
    