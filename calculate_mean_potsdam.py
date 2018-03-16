import os

import numpy as np
from libtiff import TIFF
from scipy.misc import imread

BASE_DIRECTORY = 'ISPRS_semantic_labeling_Potsdam'

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/4_Ortho_RGBIR'):
    if '.tif' in filename:
        image = TIFF.open(BASE_DIRECTORY+'/4_Ortho_RGBIR/'+filename,'r')
        image = image.read_image()
        image = image[:,:,0]
        mean.append(np.mean(image))
print('R:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/4_Ortho_RGBIR'):
    if '.tif' in filename:
        image = TIFF.open(BASE_DIRECTORY+'/4_Ortho_RGBIR/'+filename,'r')
        image = image.read_image()
        image = image[:,:,1]
        mean.append(np.mean(image))
print('G:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/4_Ortho_RGBIR'):
    if '.tif' in filename:
        image = TIFF.open(BASE_DIRECTORY+'/4_Ortho_RGBIR/'+filename,'r')
        image = image.read_image()
        image = image[:,:,2]
        mean.append(np.mean(image))
print('B:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/4_Ortho_RGBIR'):
    if '.tif' in filename:
        image = TIFF.open(BASE_DIRECTORY+'/4_Ortho_RGBIR/'+filename,'r')
        image = image.read_image()
        image = image[:,:,3]
        mean.append(np.mean(image))
print('IR:')
print(np.mean(np.array(mean)))
print()


mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/1_DSM'):
    if '.tif' in filename:
        image = imread(BASE_DIRECTORY+'/1_DSM/'+filename)
        mean.append(np.mean(image))
print('DSM:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/1_DSM_normalisation'):
    if 'lastools' in filename:
        image = imread(BASE_DIRECTORY+'/1_DSM_normalisation/'+filename)
        mean.append(np.mean(image))
print('NDSM:')
print(np.mean(np.array(mean)))
print()