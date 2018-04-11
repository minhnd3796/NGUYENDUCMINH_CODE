import os

import numpy as np
import scipy.misc as misc
from cv2 import imread, imwrite
from sys import argv

base_dir_train = "../ISPRS_semantic_labeling_Vaihingen/train"
base_dir_tiny_train = "../ISPRS_semantic_labeling_Vaihingen/tiny_train"
base_dir_validate = "../ISPRS_semantic_labeling_Vaihingen/validate"
base_dir_annotations = "../ISPRS_semantic_labeling_Vaihingen/annotations"
base_dir_top = "../ISPRS_semantic_labeling_Vaihingen/top"
base_dir_tiny_train_gt = "../ISPRS_semantic_labeling_Vaihingen/tiny_train_gt"
base_dir_train_validate_gt = "../ISPRS_semantic_labeling_Vaihingen/train_validate_gt"
image_size = 224
CROP_SIZE = 224
num_cropping_per_image = 3333
validate_image = ["top_mosaic_09cm_area7.png","top_mosaic_09cm_area17.png","top_mosaic_09cm_area23.png","top_mosaic_09cm_area37.png"]

STRIDE = int(argv[1])
def create_training_dataset_giant():
    for filename in os.listdir(base_dir_annotations):
        if filename in validate_image:
            continue
        top_image = imread(os.path.join(base_dir_top, os.path.splitext(filename)[0]+".tif"))
        annotation_image = imread(os.path.join(base_dir_annotations, filename), -1)
        width = np.shape(top_image)[1]
        height = np.shape(top_image)[0]
        i = 0
        x = 0
        while x + (CROP_SIZE - 1) <= height - 1:
            y = 0
            while y + (CROP_SIZE - 1) <= width - 1:
                print(filename, (x, y))
                top_image_cropped = top_image[x:x + image_size, y:y + image_size, :]
                imwrite(os.path.join(base_dir_train, os.path.splitext(filename)[0] + "_" + str(i) + ".tif"), top_image_cropped)
                annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
                imwrite(os.path.join(base_dir_train_validate_gt, os.path.splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
                i += 1
                y += STRIDE
            x += STRIDE
    return None


def create_validation_dataset_giant():
    for filename in validate_image:
        top_image = imread(os.path.join(base_dir_top, os.path.splitext(filename)[0] + ".tif"))
        annotation_image = imread(os.path.join(base_dir_annotations, filename), -1)
        width = np.shape(top_image)[1]
        height = np.shape(top_image)[0]
        i = 0
        x = 0
        while x + (CROP_SIZE - 1) <= height - 1:
            y = 0
            while y + (CROP_SIZE - 1) <= width - 1:
                print(filename, (x, y))
                top_image_cropped = top_image[x:x + image_size, y:y + image_size, :]
                imwrite(
                    os.path.join(base_dir_validate, os.path.splitext(filename)[0] + "_" + str(i) + ".tif"),
                    top_image_cropped)
                annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
                imwrite(os.path.join(base_dir_train_validate_gt,
                                    os.path.splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
                i += 1
                y += STRIDE
            x += STRIDE
    return None

def create_training_dataset():
    for filename in os.listdir(base_dir_annotations):
        if filename in validate_image:
            continue
        # top_image = misc.imread(os.path.join(base_dir_top, os.path.splitext(filename)[0]+".tif"))
        top_image = imread(os.path.join(base_dir_top, os.path.splitext(filename)[0]+".tif"))
        # annotation_image = misc.imread(os.path.join(base_dir_annotations, filename))
        annotation_image = imread(os.path.join(base_dir_annotations, filename), -1)
        width = np.shape(top_image)[1]
        height = np.shape(top_image)[0]
        for i in range(num_cropping_per_image):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x, y))
            top_image_cropped = top_image[x:x + image_size, y:y + image_size, :]
            # misc.imsave(os.path.join(base_dir_train, os.path.splitext(filename)[0] + "_" + str(i) + ".tif"), top_image_cropped)
            imwrite(os.path.join(base_dir_train, os.path.splitext(filename)[0] + "_" + str(i) + ".tif"), top_image_cropped)
            annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
            # misc.imsave(os.path.join(base_dir_train_validate_gt, os.path.splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
            imwrite(os.path.join(base_dir_train_validate_gt, os.path.splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
    return None


def create_validation_dataset():
    for filename in validate_image:
        # top_image = misc.imread(os.path.join(base_dir_top, os.path.splitext(filename)[0] + ".tif"))
        top_image = imread(os.path.join(base_dir_top, os.path.splitext(filename)[0] + ".tif"))
        # annotation_image = misc.imread(os.path.join(base_dir_annotations, filename))
        annotation_image = imread(os.path.join(base_dir_annotations, filename), -1)
        width = np.shape(top_image)[1]
        height = np.shape(top_image)[0]
        for i in range(num_cropping_per_image):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x, y))
            top_image_cropped = top_image[x:x + image_size, y:y + image_size, :]
            # misc.imsave(
            #     os.path.join(base_dir_validate, os.path.splitext(filename)[0] + "_" + str(i) + ".tif"),
            #     top_image_cropped)
            # annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
            # misc.imsave(os.path.join(base_dir_train_validate_gt,
            #                          os.path.splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
            imwrite(
                os.path.join(base_dir_validate, os.path.splitext(filename)[0] + "_" + str(i) + ".tif"),
                top_image_cropped)
            annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
            imwrite(os.path.join(base_dir_train_validate_gt,
                                os.path.splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
    return None

def create_tiny_training_dataset():
    for filename in os.listdir(base_dir_annotations):
        if filename in validate_image:
            continue
        top_image = misc.imread(os.path.join(base_dir_top,os.path.splitext(filename)[0]+".tif"))
        annotation_image = misc.imread(os.path.join(base_dir_annotations, filename))
        width= np.shape(top_image)[1]
        height= np.shape(top_image)[0]
        for i in range(10):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x,y))
            top_image_cropped= top_image[x:x + image_size, y:y + image_size, :]
            misc.imsave(os.path.join(base_dir_tiny_train, "tiny_"+os.path.splitext(filename)[0] + "_" + str(i) + ".tif"), top_image_cropped)
            annotation_image_cropped= annotation_image[x:x + image_size, y:y + image_size]
            misc.imsave(os.path.join(base_dir_train_validate_gt, "tiny_"+os.path.splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
    return None

create_training_dataset_giant()
create_validation_dataset_giant()
