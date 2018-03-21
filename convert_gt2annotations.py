import os

import numpy as np
from scipy import 
from PIL import Image

ground_truth_path = "../ISPRS_semantic_labeling_Vaihingen/gts_for_participants"
annotaions_path = "../ISPRS_semantic_labeling_Vaihingen/annotations"

for filename in os.listdir(ground_truth_path):
    image= misc.imread(os.path.join(ground_truth_path, filename))
    annotation_image=np.zeros((np.shape(image)[0], np.shape(image)[1]))
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
                annotation_image[i, j] = 5
    annotation_filename = os.path.splitext(filename)[0]
    print(">> Processing", annotation_image)
    # misc.toimage(annotation_image, high = high, low=0).save(os.path.join(annotaions_path, annotation_filename + ".png"))
    Image.fromarray(annotation_image).save(os.path.join(annotaions_path, annotation_filename + ".png"))
print("Done!")
