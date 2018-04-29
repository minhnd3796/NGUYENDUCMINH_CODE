import os
import numpy as np
import scipy.misc as misc
from cv2 import imread, imwrite
# from sys import argv
from os.path import exists, join, splitext
from os import listdir, mkdir

base_dir_train = "../ISPRS_semantic_labeling_Vaihingen/train_3channels"
base_dir_validate = "../ISPRS_semantic_labeling_Vaihingen/validate_3channels"
base_dir_train_validate_gt = "../ISPRS_semantic_labeling_Vaihingen/train_validate_gt_3channels"

base_dir_top = "../ISPRS_semantic_labeling_Vaihingen/top"
base_dir_annotations = "../ISPRS_semantic_labeling_Vaihingen/annotations"

base_dir_tiny_train = "../ISPRS_semantic_labeling_Vaihingen/tiny_train"
base_dir_tiny_train_gt = "../ISPRS_semantic_labeling_Vaihingen/tiny_train_gt"

image_size = 224
CROP_SIZE = 224
num_cropping_per_image = 4096
validate_image = ["top_mosaic_09cm_area7.png","top_mosaic_09cm_area17.png","top_mosaic_09cm_area23.png","top_mosaic_09cm_area37.png"]

def create_training_dataset():
    if not exists(base_dir_train):
        mkdir(base_dir_train)
    if not exists(base_dir_train_validate_gt):
        mkdir(base_dir_train_validate_gt)
    for filename in os.listdir(base_dir_annotations):
        if filename in validate_image:
            continue
        # top_image = misc.imread(join(base_dir_top, splitext(filename)[0]+".tif"))
        top_image = imread(join(base_dir_top, splitext(filename)[0]+".tif"))
        # annotation_image = misc.imread(join(base_dir_annotations, filename))
        annotation_image = imread(join(base_dir_annotations, filename), -1)
        width = np.shape(top_image)[1]
        height = np.shape(top_image)[0]
        for i in range(num_cropping_per_image):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x, y))
            top_image_cropped = top_image[x:x + image_size, y:y + image_size, :]
            # misc.imsave(join(base_dir_train, splitext(filename)[0] + "_" + str(i) + ".tif"), top_image_cropped)
            imwrite(join(base_dir_train, splitext(filename)[0] + "_" + str(4 * i) + ".tif"), top_image_cropped)
            imwrite(join(base_dir_train, splitext(filename)[0] + "_" + str(4 * i + 1) + ".tif"), np.fliplr(top_image_cropped))
            imwrite(join(base_dir_train, splitext(filename)[0] + "_" + str(4 * i + 2) + ".tif"), np.flipud(top_image_cropped))
            imwrite(join(base_dir_train, splitext(filename)[0] + "_" + str(4 * i + 3) + ".tif"), np.flipud(np.fliplr(top_image_cropped)))
            annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
            # misc.imsave(join(base_dir_train_validate_gt, splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
            imwrite(join(base_dir_train_validate_gt, splitext(filename)[0] + "_" + str(4 * i) + ".png"), annotation_image_cropped)
            imwrite(join(base_dir_train_validate_gt, splitext(filename)[0] + "_" + str(4 * i + 1) + ".png"), np.fliplr(annotation_image_cropped))
            imwrite(join(base_dir_train_validate_gt, splitext(filename)[0] + "_" + str(4 * i + 2) + ".png"), np.flipud(annotation_image_cropped))
            imwrite(join(base_dir_train_validate_gt, splitext(filename)[0] + "_" + str(4 * i + 3) + ".png"), np.flipud(np.fliplr(annotation_image_cropped)))
    return None


def create_validation_dataset():
    if not exists(base_dir_validate):
        mkdir(base_dir_validate)
    for filename in validate_image:
        # top_image = misc.imread(join(base_dir_top, splitext(filename)[0] + ".tif"))
        top_image = imread(join(base_dir_top, splitext(filename)[0] + ".tif"))
        # annotation_image = misc.imread(join(base_dir_annotations, filename))
        annotation_image = imread(join(base_dir_annotations, filename), -1)
        width = np.shape(top_image)[1]
        height = np.shape(top_image)[0]
        for i in range(num_cropping_per_image):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x, y))
            top_image_cropped = top_image[x:x + image_size, y:y + image_size, :]
            # misc.imsave(
            #     join(base_dir_validate, splitext(filename)[0] + "_" + str(i) + ".tif"),
            #     top_image_cropped)
            # annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
            # misc.imsave(join(base_dir_train_validate_gt,
            #                          splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
            imwrite(
                join(base_dir_validate, splitext(filename)[0] + "_" + str(i) + ".tif"),
                top_image_cropped)
            annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
            imwrite(join(base_dir_train_validate_gt,
                                splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
    return None

def create_tiny_training_dataset():
    for filename in os.listdir(base_dir_annotations):
        if filename in validate_image:
            continue
        top_image = misc.imread(join(base_dir_top,splitext(filename)[0]+".tif"))
        annotation_image = misc.imread(join(base_dir_annotations, filename))
        width= np.shape(top_image)[1]
        height= np.shape(top_image)[0]
        for i in range(10):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x,y))
            top_image_cropped= top_image[x:x + image_size, y:y + image_size, :]
            misc.imsave(join(base_dir_tiny_train, "tiny_"+splitext(filename)[0] + "_" + str(i) + ".tif"), top_image_cropped)
            annotation_image_cropped= annotation_image[x:x + image_size, y:y + image_size]
            misc.imsave(join(base_dir_train_validate_gt, "tiny_"+splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
    return None

if __name__=="__main__":
    np.random.seed(3796)
    create_training_dataset()
    create_validation_dataset()
