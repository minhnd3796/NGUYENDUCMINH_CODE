from __future__ import print_function

# from scipy.misc import imread, imsave
from cv2 import imread, imwrite
import numpy as np
from os import listdir
from os.path import splitext

file = listdir('../ISPRS_semantic_labeling_Vaihingen/annotations/')
for f in file:
    predict_annotation_image = imread('../ISPRS_semantic_labeling_Vaihingen/annotations/' + f, -1)
    height, width = predict_annotation_image.shape
    output_image = np.zeros((height, width, 3))
    print('>> Processing', f)
    for i in range(height):
        for j in range(width):
            if predict_annotation_image[i,j]==0:
                output_image[i,j,:]=[255,255,255]
            elif predict_annotation_image[i,j]==1:
                output_image[i,j,:]=[0,0,255]
            elif predict_annotation_image[i,j]==3:
                output_image[i,j,:]=[0,255,255]
            elif predict_annotation_image[i,j]==4:
                output_image[i,j,:]=[0,255,0]
            elif predict_annotation_image[i,j]==5:
                output_image[i,j,:]=[255,255,0]
            elif predict_annotation_image[i,j]==2:
                output_image[i,j,:]=[255,0,0]
    # imsave('../ISPRS_semantic_labeling_Vaihingen/annotations_back2_gts/' + splitext(f)[0] + '.tif', output_image)
    imwrite('../ISPRS_semantic_labeling_Vaihingen/annotations_back2_gts/' + splitext(f)[0] + '.tif', output_image)