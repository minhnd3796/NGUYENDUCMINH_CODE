import os

import numpy as np
from scipy.misc import imread

BASE_DIRECTORY = 'ISPRS_semantic_labeling_Vaihingen'

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/top'):
    image = imread(BASE_DIRECTORY+'/top/'+filename)
    image = image[:,:,0]
    mean.append(np.mean(image))
print('IR:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/top'):
    image = imread(BASE_DIRECTORY+'/top/'+filename)
    image = image[:,:,1]
    mean.append(np.mean(image))
print('R:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/top'):
    image = imread(BASE_DIRECTORY+'/top/'+filename)
    image = image[:,:,2]
    mean.append(np.mean(image))
print('G:')
print(np.mean(np.array(mean)))
print()


mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/A'):
    image = imread(BASE_DIRECTORY+'/A/'+filename)
    mean.append(np.mean(image))
print('A:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/dsm'):
    image = imread(BASE_DIRECTORY+'/dsm/'+filename)
    mean.append(np.mean(image))
print('DSM:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/B'):
    image = imread(BASE_DIRECTORY+'/B/'+filename)
    mean.append(np.mean(image))
print('B:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/azi'):
    image = imread(BASE_DIRECTORY+'/azi/'+filename)
    mean.append(np.mean(image))
print('azi:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/L'):
    image = imread(BASE_DIRECTORY+'/L/'+filename)
    mean.append(np.mean(image))
print('L:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/ele'):
    image = imread(BASE_DIRECTORY+'/ele/'+filename)
    mean.append(np.mean(image))
print('ele:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/entpy'):
    image = imread(BASE_DIRECTORY+'/entpy/'+filename)
    mean.append(np.mean(image))
print('entpy:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/entpy2'):
    image = imread(BASE_DIRECTORY+'/entpy2/'+filename)
    mean.append(np.mean(image))
print('entpy2:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/ndsm'):
    image = imread(BASE_DIRECTORY+'/ndsm/'+filename)
    mean.append(np.mean(image))
print('ndsm:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/ndvi'):
    image = imread(BASE_DIRECTORY+'/ndvi/'+filename)
    mean.append(np.mean(image))
print('ndvi:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/sat'):
    image = imread(BASE_DIRECTORY+'/sat/'+filename)
    mean.append(np.mean(image))
print('sat:')
print(np.mean(np.array(mean)))
print()

mean =[]
for filename in os.listdir(BASE_DIRECTORY+'/texton'):
    image = imread(BASE_DIRECTORY+'/texton/'+filename)
    mean.append(np.mean(image))
print('texton:')
print(np.mean(np.array(mean)))

# A:
# 106.314329243
#
# DSM:
# 283.307
#
# B:
# 109.260369903
#
# azi:
# 124.171918054
#
# L:
# 100.699252985
#
# ele:
# 182.615729022
#
# entpy2:
# 84.3529895303
#
# ndsm:
# 30.6986130799
#
# ndvi:
# 66.8837693324
#
# sat:
# 98.6030061849
#
# texton:
# 133.955897217