import os
import numpy as np
import scipy.misc as misc
from cv2 import imread, imwrite
# from sys import argv
from os.path import exists, join, splitext
from os import listdir, mkdir

base_dir_train = "../ISPRS_semantic_labeling_Potsdam/training_set_submission"
base_dir_validate = "../ISPRS_semantic_labeling_Potsdam/validation_set"
base_dir_train_validate_gt = "../ISPRS_semantic_labeling_Potsdam/ground_truths_submission"
base_dir_top = "../ISPRS_semantic_labeling_Potsdam/npy_6_channel"
base_dir_annotations = "../ISPRS_semantic_labeling_Potsdam/annotations"

test_dir_train = "../ISPRS_semantic_labeling_Potsdam/training_set_test"
test_dir_validate = "../ISPRS_semantic_labeling_Potsdam/validation_set_test"
test_dir_train_validate_gt = "../ISPRS_semantic_labeling_Potsdam/ground_truths_test"
test_dir_top = "../ISPRS_semantic_labeling_Potsdam/npy_6_channel_test"
test_dir_annotations = "../ISPRS_semantic_labeling_Potsdam/annotations_test"

image_size = 224
CROP_SIZE = 224
num_cropping_per_image = 4096
num_cropping_per_test_image = 32
# validate_image = ["top_potsdam_2_11_label.png","top_potsdam_3_12_label.png","top_potsdam_4_10_label.png","top_potsdam_6_11_label.png","top_potsdam_7_12_label.png"]
validate_image = []
validate_image_test = ["top_potsdam_2_11_label.png"]

def create_training_dataset():
    if not exists(base_dir_train):
        mkdir(base_dir_train)
    if not exists(base_dir_train_validate_gt):
        mkdir(base_dir_train_validate_gt)
    for filename in os.listdir(base_dir_annotations):
        if filename in validate_image:
            continue
        top_image = np.load(join(base_dir_top, splitext(filename)[0].replace('label','RGBIR')+".npy"))
        annotation_image = imread(join(base_dir_annotations, filename), -1)
        width = np.shape(top_image)[1]
        height = np.shape(top_image)[0]
        for i in range(num_cropping_per_image):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x, y))
            top_image_cropped = top_image[x:x + image_size, y:y + image_size, :]
            np.save(join(base_dir_train, splitext(filename)[0] + "_" + str(4 * i) + ".npy"), top_image_cropped)
            np.save(join(base_dir_train, splitext(filename)[0] + "_" + str(4 * i + 1) + ".npy"), np.fliplr(top_image_cropped))
            np.save(join(base_dir_train, splitext(filename)[0] + "_" + str(4 * i + 2) + ".npy"), np.flipud(top_image_cropped))
            np.save(join(base_dir_train, splitext(filename)[0] + "_" + str(4 * i + 3) + ".npy"), np.flipud(np.fliplr(top_image_cropped)))
            annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
            imwrite(join(base_dir_train_validate_gt, splitext(filename)[0] + "_" + str(4 * i) + ".png"), annotation_image_cropped)
            imwrite(join(base_dir_train_validate_gt, splitext(filename)[0] + "_" + str(4 * i + 1) + ".png"), np.fliplr(annotation_image_cropped))
            imwrite(join(base_dir_train_validate_gt, splitext(filename)[0] + "_" + str(4 * i + 2) + ".png"), np.flipud(annotation_image_cropped))
            imwrite(join(base_dir_train_validate_gt, splitext(filename)[0] + "_" + str(4 * i + 3) + ".png"), np.flipud(np.fliplr(annotation_image_cropped)))
    return None

def create_validation_dataset():
    if not exists(base_dir_validate):
        mkdir(base_dir_validate)
    for filename in validate_image:
        top_image = np.load(join(base_dir_top, splitext(filename)[0].replace('label','RGBIR')+".npy"))
        annotation_image = imread(join(base_dir_annotations, filename), -1)
        width = np.shape(top_image)[1]
        height = np.shape(top_image)[0]
        for i in range(num_cropping_per_image):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x, y))
            top_image_cropped = top_image[x:x + image_size, y:y + image_size, :]
            np.save(
                join(base_dir_validate, splitext(filename)[0] + "_" + str(i) + ".npy"),
                top_image_cropped)
            annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
            imwrite(join(base_dir_train_validate_gt,
                                splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
    return None

def create_training_test_dataset():
    if not exists(test_dir_train):
        mkdir(test_dir_train)
    if not exists(test_dir_train_validate_gt):
        mkdir(test_dir_train_validate_gt)
    for filename in os.listdir(test_dir_annotations):
        if filename in validate_image_test:
            continue
        top_image = np.load(join(test_dir_top, splitext(filename)[0].replace('label','RGBIR')+".npy"))
        annotation_image = imread(join(test_dir_annotations, filename), -1)
        width = np.shape(top_image)[1]
        height = np.shape(top_image)[0]
        for i in range(num_cropping_per_test_image):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x, y))
            top_image_cropped = top_image[x:x + image_size, y:y + image_size, :]
            np.save(join(test_dir_train, splitext(filename)[0] + "_" + str(4 * i) + ".npy"), top_image_cropped)
            np.save(join(test_dir_train, splitext(filename)[0] + "_" + str(4 * i + 1) + ".npy"), np.fliplr(top_image_cropped))
            np.save(join(test_dir_train, splitext(filename)[0] + "_" + str(4 * i + 2) + ".npy"), np.flipud(top_image_cropped))
            np.save(join(test_dir_train, splitext(filename)[0] + "_" + str(4 * i + 3) + ".npy"), np.flipud(np.fliplr(top_image_cropped)))
            annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
            imwrite(join(test_dir_train_validate_gt, splitext(filename)[0] + "_" + str(4 * i) + ".png"), annotation_image_cropped)
            imwrite(join(test_dir_train_validate_gt, splitext(filename)[0] + "_" + str(4 * i + 1) + ".png"), np.fliplr(annotation_image_cropped))
            imwrite(join(test_dir_train_validate_gt, splitext(filename)[0] + "_" + str(4 * i + 2) + ".png"), np.flipud(annotation_image_cropped))
            imwrite(join(test_dir_train_validate_gt, splitext(filename)[0] + "_" + str(4 * i + 3) + ".png"), np.flipud(np.fliplr(annotation_image_cropped)))
    return None

def create_validation_test_dataset():
    if not exists(test_dir_validate):
        mkdir(test_dir_validate)
    for filename in validate_image_test:
        top_image = np.load(join(test_dir_top, splitext(filename)[0].replace('label','RGBIR')+".npy"))
        annotation_image = imread(join(test_dir_annotations, filename), -1)
        width = np.shape(top_image)[1]
        height = np.shape(top_image)[0]
        for i in range(num_cropping_per_test_image):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x, y))
            top_image_cropped = top_image[x:x + image_size, y:y + image_size, :]
            np.save(
                join(test_dir_validate, splitext(filename)[0] + "_" + str(i) + ".npy"),
                top_image_cropped)
            annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
            imwrite(join(test_dir_train_validate_gt,
                                splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
    return None

if __name__=="__main__":
    np.random.seed(3796)
    create_training_dataset()
    # create_validation_dataset()
    """ create_training_test_dataset()
    create_validation_test_dataset() """
