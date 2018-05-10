import os

import numpy as np
from libtiff import TIFF
from cv2 import imread

BASE_DIRECTORY = '../ISPRS_semantic_labeling_Potsdam'

""" mean =[]
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
print() """


mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/1_DSM'):
    if '.tif' in filename:
        image = TIFF.open(BASE_DIRECTORY+'/1_DSM/'+filename,'r').read_image()
        mean.append(np.mean(image))
print('DSM:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/1_DSM_normalisation'):
    if 'lastools' in filename:
        image = imread(BASE_DIRECTORY+'/1_DSM_normalisation/'+filename, -1)
        mean.append(np.mean(image))
print('NDSM:')
print(np.mean(np.array(mean)))
print()

# R: 86.55175021564328
# G: 92.54522770394738
# B: 85.91596489181288
# IR: 97.639895122076
# DSM: 37.69138
# nDSM: 45.548982715963994