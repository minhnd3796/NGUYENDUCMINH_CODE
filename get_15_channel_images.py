from scipy.io import loadmat
from cv2 import imread
from os.path import join, splitext, exists
from os import listdir, mkdir
import numpy as np

file_indices = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38,
              1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37,
              11, 15, 28, 30, 34]

top_dir = "../ISPRS_semantic_labeling_Vaihingen/top"
ndsm_dir = "../ISPRS_semantic_labeling_Vaihingen/ndsm"
dsm_dir = "../ISPRS_semantic_labeling_Vaihingen/dsm"
nine_feature_mat_dir = "../ISPRS_semantic_labeling_Vaihingen/9_feature_mat"
filename = "top_mosaic_09cm_area"

def get_15_channel_array(file_index):
    top_image_name = filename + str(file_index) + ".tif"
    top_image = imread(join(top_dir, top_image_name))

    dsm_image_name = filename.replace('top_mosaic', 'dsm').replace('area', 'matching_area')
    dsm_image = imread(join(dsm_dir, dsm_image_name), -1)

    ndsm_image_name = dsm_image_name.replace('.tif', '') + "_normalized.jpg"
    ndsm_image = imread(join(ndsm_dir, ndsm_image_name), -1)

    texton_mat = loadmat(join("../ISPRS_semantic_labeling_Vaihingen/texton_mat", "texton" + str(file_index) + ".mat"))
    texton_image = texton_mat['texton']
    
    feature_dict = loadmat(join(nine_feature_mat_dir, "feat" + str(file_index) + ".mat"))
    A_image = feature_dict['a']
    ele_image = feature_dict['ele']
    L_image = feature_dict['l']
    azi_image = feature_dict['azi']
    B_image = feature_dict['b']
    entpy_image = feature_dict['entpy']
    sat_image = feature_dict['sat']
    entpy2_image = feature_dict['entpy2']
    ndvi_image = feature_dict['ndvi']

    return np.concatenate((top_image, ndsm_image, dsm_image, A_image, azi_image, B_image, ele_image,
                           entpy_image, entpy2_image, L_image, ndvi_image, sat_image, texton_image), axis=2).astype(dtype=np.float16)