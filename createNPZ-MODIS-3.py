# -*- coding: utf-8 -*-
"""
Created on Sun May  8 04:46:43 2022

@author: xuan
"""
import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
import cv2
import sys
import numpy as np
from os import listdir

def resizecloud(array):
    return cv2.resize(array, (int(1354/2),int(2030/2)), interpolation = cv2.INTER_NEAREST)

def showimg(img):
    plt.imshow(img, cmap='gray')
    plt.show()


filepath080=os.listdir('D:/Research/Data/Combined/dec-15-2020/')
print(filepath080)

# test the first file
tarr=np.load('D:/Research/Data/Combined/dec-15-2020/'+filepath080[0])
print(tarr[:,:,0].shape)
resizecloud(tarr[:,:,0])
showimg(tarr[:,:,4])
showimg(resizecloud(tarr[:,:,4]))
narr = resizecloud(tarr[:,:,0])
print(narr.shape)


# split into data and label
data = np.zeros((len(filepath080),1015, 677, 4))
label = np.zeros((len(filepath080),1015, 677))
for i in range(len(filepath080)):
    oridata = np.load('D:/Research/Data/Combined/dec-15-2020/'+filepath080[i])
    for j in range(4):
        data[i,:,:,j]=resizecloud(oridata[:,:,j])
        
    print('i, max_data = ', i, np.max(data[i,:,:,:]))
    label[i,:,:] = resizecloud(oridata[:,:,4])
 
print('max_data = ', np.max(data))
print('max_label = ', np.max(label))

# show one image   
showimg(data[10,:,:,0])
showimg(data[10,:,:,1])
showimg(data[10,:,:,2])
showimg(data[10,:,:,3])
showimg(label[10,:,:])


# save as an npz file
np.savez('D:/Research/Data/Combined/dec-15-2020.npz', data=data, label=label)

 


