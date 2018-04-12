from cv2 import imread
import numpy as np
from os.path import join
from sys import argv

validation_image = ['top_mosaic_09cm_area7', 'top_mosaic_09cm_area17', 'top_mosaic_09cm_area23',
                  'top_mosaic_09cm_area37']
training_image = ['top_mosaic_09cm_area1', 'top_mosaic_09cm_area3', 'top_mosaic_09cm_area5',
                  'top_mosaic_09cm_area11', 'top_mosaic_09cm_area13', 'top_mosaic_09cm_area15',
                  'top_mosaic_09cm_area21', 'top_mosaic_09cm_area26', 'top_mosaic_09cm_area28',
                  'top_mosaic_09cm_area30', 'top_mosaic_09cm_area32', 'top_mosaic_09cm_area34']

num_training_crops = 0
num_validation_crops = 0
CROP_SIZE = 224
STRIDE = int(argv[1])

for img_name in validation_image:
    img = imread(join('../ISPRS_semantic_labeling_Vaihingen/annotations_back2_gts', img_name + '.tif'))
    print("Processing", img_name)
    # num_validation_crops += (img.shape[0] - CROP_SIZE + 1) * (img.shape[1] - CROP_SIZE + 1)
    i = 0
    while i + (CROP_SIZE - 1) <= img.shape[0] - 1:
        j = 0
        while j + (CROP_SIZE - 1) <= img.shape[1] - 1:
            num_validation_crops += 1
            j += STRIDE
        i += STRIDE

for img_name in training_image:
    img = imread(join('../ISPRS_semantic_labeling_Vaihingen/annotations_back2_gts', img_name + '.tif'))
    print("Processing", img_name)
    # num_training_crops += (img.shape[0] - CROP_SIZE + 1) * (img.shape[1] - CROP_SIZE + 1)
    i = 0
    while i + (CROP_SIZE - 1) <= img.shape[0] - 1:
        j = 0
        while j + (CROP_SIZE - 1) <= img.shape[1] - 1:
            num_training_crops += 1
            j += STRIDE
        i += STRIDE

print("No. of training images:", num_training_crops)
print("Training size:", num_training_crops * 504271.0 / 1024.0 / 1024.0 / 1024.0)
print()
print("No. of validation images:", num_validation_crops)
print("Validation size:", num_validation_crops * 504271.0 / 1024.0 / 1024.0 / 1024.0)
print()
print("Total:", num_training_crops + num_validation_crops)
print("Total size:", (num_training_crops + num_validation_crops) * 504271.0 / 1024.0 / 1024.0 / 1024.0)