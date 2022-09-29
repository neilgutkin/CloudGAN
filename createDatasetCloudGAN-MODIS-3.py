from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt

import os
import pathlib
import time
import datetime

from os import  listdir
import pandas as pd
import cv2

from pyhdf.SD import SD, SDC

GTnames = listdir('D:/Research/Data/CloudMask/dec-15-2020')

for a in range(len(GTnames)):
# for a in range(1):

    
    
    gtfilename = GTnames[a]
    
    
    nc_gt = NetCDFFile('D:/Research/Data/CloudMask/dec-15-2020/'+gtfilename)
    
    cloud_mask = nc_gt.groups['geophysical_data'].variables['Integer_Cloud_Mask'][:]
    cloud_mask = (np.logical_and(cloud_mask != -1, cloud_mask <= 1)).astype(float)
    cloud_mask = (cv2.flip(cloud_mask, -1)).astype(float)
    cloud_mask = pd.DataFrame(cloud_mask).interpolate( limit_direction='both').to_numpy()
    
    # cloud_mask = cv2.flip(cloud_mask, 1)
    
    datafilename = 'MYD021KM.A2020350.' + gtfilename[30:34] + '.hdf'
    
    print('D:/Research/Data/MODIS/dec-15-2020/'+datafilename)
    
    the_file = SD('D:/Research/Data/MODIS/dec-15-2020/'+datafilename, SDC.READ)
    
    wave_data = the_file.select('EV_250_Aggr1km_RefSB') # select sds
    R = wave_data[0,:,:]
    R = cv2.flip(R, -1)
    NIR = wave_data[1,:,:]
    NIR = cv2.flip(NIR, -1)
    # print(R.shape, NIR.shape, cloud_mask.shape)
    R=pd.DataFrame(R).interpolate( limit_direction='both').to_numpy()
    NIR=pd.DataFrame(NIR).interpolate( limit_direction='both').to_numpy()
    
    wave_data = the_file.select('EV_500_Aggr1km_RefSB') # select sds
    B = wave_data[0,:,:]
    B = cv2.flip(B, -1)
    G = wave_data[1,:,:]
    G = cv2.flip(G, -1)
    # print(B.shape, G.shape, cloud_mask.shape)
    B=pd.DataFrame(B).interpolate( limit_direction='both').to_numpy()
    G=pd.DataFrame(G).interpolate( limit_direction='both').to_numpy()
    
    # print(B.shape, G.shape, R.shape, NIR.shape, cloud_mask.shape)
    
    datapoint = np.stack((B,G,R,NIR,cloud_mask),axis=-1)
    np.save('D:/Research/Data/Combined/dec-15-2020/'+gtfilename[30:34]+'.npy', datapoint)
    print(gtfilename[30:34]+' saved')
    
    #'''
    plt.subplot(1, 5, 1)
    plt.title(gtfilename[30:34])
    plt.imshow(B, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 5, 2)
    plt.imshow(G, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 5, 3)
    plt.imshow(R, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 5, 4)
    plt.imshow(NIR, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 5, 5)
    plt.imshow(cloud_mask, cmap='gray')
    plt.axis('off')
    plt.show()
    #'''