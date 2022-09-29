"""
Displays MODIS/CloudMask images stored in npy files

@author Neil Gutkin
"""

import numpy as np
import matplotlib.pyplot as plt
from os import listdir

directory = 'D:/Research/Data/Combined/dec-15-2020/'
file_names = listdir(directory)

for name in file_names:
    data = np.load(directory + name)

    B = data[:,:,0]
    G = data[:,:,1]
    R = data[:,:,2]
    NIR = data[:,:,3]
    cloud_mask = data[:,:,4]

    fig = plt.figure(figsize=(16,9))
    plt.subplot(1, 5, 1)
    plt.title(name)
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