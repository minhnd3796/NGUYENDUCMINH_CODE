import os

import numpy as np
from scipy import misc
from cv2 import imread, imwrite

ground_truth_path = "../ISPRS_semantic_labeling_Potsdam/5_Labels_for_participants"
annotaions_path = "../ISPRS_semantic_labeling_Potsdam/annotations"

for filename in os.listdir(ground_truth_path):
    print(">> Processing", filename)
    image = imread(os.path.join(ground_truth_path, filename))
    annotation_image = np.zeros((np.shape(image)[0], np.shape(image)[1]))
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            if np.array_equal(image[i, j, :], np.array([255, 255, 255])):
                # Impervious surfaces (RGB: 255, 255, 255)
                annotation_image[i, j] = 0
            elif np.array_equal(image[i,j,:],np.array([0, 0, 255])):
                # Building (RGB: 0, 0, 255)
                annotation_image[i, j] = 1
            elif np.array_equal(image[i,j,:],np.array([0, 255, 255])):
                # Low vegetation (RGB: 0, 255, 255)
                annotation_image[i, j] = 2
            elif np.array_equal(image[i,j,:],np.array([0, 255, 0])):
                # Tree (RGB: 0, 255, 0)
                annotation_image[i, j] = 3
            elif np.array_equal(image[i,j,:],np.array([255, 255, 0])):
                # Car (RGB: 255, 255, 0)
                annotation_image[i, j] = 4
            else:
                # Clutter/background (RGB: 255, 0, 0)
                # high = 5
                annotation_image[i, j] = 5
    annotation_filename = os.path.splitext(filename)[0]
    imwrite(os.path.join(annotaions_path, annotation_filename + ".png"), annotation_image)
print('Done!')

""" import os

import numpy as np
from scipy import misc

ground_truth_path="ISPRS_semantic_labeling_Potsdam/5_Labels_for_participants"
annotaions_path="ISPRS_semantic_labeling_Potsdam/annotations"
high=4
for filename in os.listdir(ground_truth_path):
    if '.tif' in filename:
        image= misc.imread(os.path.join(ground_truth_path,filename))
        annotation_image=np.zeros((np.shape(image)[0],np.shape(image)[1]))
        for i in range(np.shape(image)[0]):
            print(i)
            for j in range(np.shape(image)[1]):
                if np.array_equal(image[i,j,:],np.array([255,255,255])):
                    annotation_image[i,j]=0
                elif np.array_equal(image[i,j,:],np.array([0,0,255])):
                    annotation_image[i, j] = 1
                elif np.array_equal(image[i,j,:],np.array([0,255,255])):
                    annotation_image[i, j] = 2
                elif np.array_equal(image[i,j,:],np.array([0,255,0])):
                    annotation_image[i, j] = 3
                elif np.array_equal(image[i,j,:],np.array([255,255,0])):
                    annotation_image[i, j] = 4
                else:
                    high=5
                    annotation_image[i,j]=5
        annotation_filename= os.path.splitext(filename)[0]
        print(np.shape(annotation_image))
        #misc.imsave(os.path.join(annotaions_path,annotation_filename+".jpg"),annotation_image)
        misc.toimage(annotation_image,high=high,low=0).save(os.path.join(annotaions_path,annotation_filename+".png"))
        high=4
print("Done!") """
