import os

import numpy as np
import scipy.misc as misc

base_dir_train = "ISPRS_semantic_labeling_Vaihingen/train"
base_dir_tiny_train = "ISPRS_semantic_labeling_Vaihingen/tiny_train"
base_dir_validate = "ISPRS_semantic_labeling_Vaihingen/validate"
base_dir_annotations = "ISPRS_semantic_labeling_Vaihingen/annotations"
base_dir_top = "ISPRS_semantic_labeling_Vaihingen/top"
base_dir_tiny_train_gt = "ISPRS_semantic_labeling_Vaihingen/tiny_train_gt"
base_dir_train_validate_gt = "ISPRS_semantic_labeling_Vaihingen/train_validate_gt"
image_size = 224
num_cropping_per_image = 3
validate_image = ["top_mosaic_09cm_area7.png","top_mosaic_09cm_area17.png","top_mosaic_09cm_area23.png","top_mosaic_09cm_area37.png"]


def create_training_dataset():
    for filename in os.listdir(base_dir_annotations):
        if filename in validate_image:
            continue
        top_image = misc.imread(os.path.join(base_dir_top,os.path.splitext(filename)[0]+".tif"))
        annotation_image = misc.imread(os.path.join(base_dir_annotations, filename))
        width= np.shape(top_image)[1]
        height= np.shape(top_image)[0]
        for i in range(num_cropping_per_image):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x,y))
            top_image_cropped= top_image[x:x + image_size, y:y + image_size, :]
            misc.imsave(os.path.join(base_dir_train, os.path.splitext(filename)[0] + "_" + str(i) + ".tif"), top_image_cropped)
            annotation_image_cropped= annotation_image[x:x + image_size, y:y + image_size]
            misc.imsave(os.path.join(base_dir_train_validate_gt, os.path.splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
    return None


def create_validation_dataset():
    for filename in validate_image:
        top_image = misc.imread(os.path.join(base_dir_top, os.path.splitext(filename)[0] + ".tif"))
        annotation_image = misc.imread(os.path.join(base_dir_annotations, filename))
        width = np.shape(top_image)[1]
        height = np.shape(top_image)[0]
        for i in range(num_cropping_per_image):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x, y))
            top_image_cropped = top_image[x:x + image_size, y:y + image_size, :]
            misc.imsave(
                os.path.join(base_dir_validate, os.path.splitext(filename)[0] + "_" + str(i) + ".tif"),
                top_image_cropped)
            annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
            misc.imsave(os.path.join(base_dir_train_validate_gt,
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

create_training_dataset()
create_tiny_training_dataset()
create_validation_dataset()