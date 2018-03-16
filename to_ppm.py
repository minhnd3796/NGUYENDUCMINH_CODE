import numpy as np
from libtiff import TIFF

image = TIFF.open('ISPRS_semantic_labeling_Potsdam/1_DSM/dsm_potsdam_03_13.tif','r')
image = image.read_image()


print(image.dtype)
column = image[:,5998]
print(np.shape(column))
column = np.expand_dims(column,axis=1)
image = np.concatenate((image,column),axis=1)
print(image.dtype)
print(np.shape(image))
image_write = TIFF.open('dsm_potsdam_03_13.tif','w')
image_write.write_image(image)
image_write.close()
print('done')



#image = imread('ISPRS_semantic_labeling_Potsdam/1_DSM_normalisation/dsm_potsdam_03_13_normalized_lastools.jpg')
#imsave('dsm_potsdam_03_13.tif',image)