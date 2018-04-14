from __future__ import print_function

# from scipy.misc import imread, imsave
from cv2 import imread, imwrite
import numpy as np
from os import listdir
from os.path import splitext

# png_dir = '../ISPRS_semantic_labeling_Vaihingen/annotations/'
png_dir = '../logs-resnet101/'
out_dir = '../ISPRS_semantic_labeling_Vaihingen/annotations_back2_gts/'

file = listdir(png_dir)
for f in file:
    if f[:3] == 'gt_' or f[:5] == 'pred_':
        predict_annotation_image = imread(png_dir + f, -1)
        height, width = predict_annotation_image.shape
        output_image = np.zeros((height, width, 3))
        print('>> Processing', f)
        for i in range(height):
            for j in range(width):
                if predict_annotation_image[i, j] == 0:
                    output_image[i, j, :] = [255, 255, 255]
                elif predict_annotation_image[i, j] == 1:
                    output_image[i, j, :] = [0, 0, 255]
                elif predict_annotation_image[i, j] == 2:
                    output_image[i, j, :] = [0, 255, 255]
                elif predict_annotation_image[i, j] == 3:
                    output_image[i, j, :] = [0, 255, 0]
                elif predict_annotation_image[i, j] == 4:
                    output_image[i, j, :] = [255, 255, 0]
                elif predict_annotation_image[i, j] == 5:
                    output_image[i, j, :] = [255, 0, 0]
        # imwrite(out_dir + splitext(f)[0] + '.tif', output_image)
        imwrite(png_dir + splitext(f)[0] + '.tif', output_image)


    
