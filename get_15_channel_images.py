from scipy.io import loadmat
import cv2
from os.path import join, splitext, exists
from os import listdir, mkdir
import numpy as np
from libtiff import TIFF

file_indices = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38,
                1, 3, 5, 7, 13, 17, 21, 23, 26, 32, 37,
                11, 15, 28, 30, 34]
# file_indices = [11]

top_dir = "../ISPRS_semantic_labeling_Vaihingen/top"
ndsm_dir = "../ISPRS_semantic_labeling_Vaihingen/ndsm"
dsm_dir = "../ISPRS_semantic_labeling_Vaihingen/dsm"
nine_feature_mat_dir = "../ISPRS_semantic_labeling_Vaihingen/9_feature_mat"
filename = "top_mosaic_09cm_area"

def get_15_channel_array(file_index):
    top_image_name = filename + str(file_index) + ".tif"
    top_image = cv2.imread(join(top_dir, top_image_name))
    # print("Top:", top_image.shape)

    dsm_image_name = filename.replace('top_mosaic', 'dsm').replace('area', 'matching_area') + str(file_index) + '.tif'
    # dsm_image = np.expand_dims(cv2.imread(join(dsm_dir, dsm_image_name), -1), axis=2)
    dsm_image = np.expand_dims(TIFF.open(join(dsm_dir, dsm_image_name), 'r').read_image(), axis=2)
    
    # print(join(dsm_dir, dsm_image_name))
    # print("DSM:", dsm_image.shape)

    ndsm_image_name = dsm_image_name.replace('.tif', '') + "_normalized.jpg"
    ndsm_image = np.expand_dims(cv2.imread(join(ndsm_dir, ndsm_image_name), -1), axis=2)
    # print(join(ndsm_dir, ndsm_image_name))
    # print("nDSM:", ndsm_image.shape)

    texton_mat = loadmat(join("../ISPRS_semantic_labeling_Vaihingen/texton_mat", "texton" + str(file_index) + ".mat"))
    texton_image = texton_mat['texton'].reshape((top_image.shape[1], top_image.shape[0]))
    texton_image = np.expand_dims(np.transpose(texton_image), axis=2)
    # print("Texton:", texton_image.shape)

    feature_dict = loadmat(join(nine_feature_mat_dir, "feat" + str(file_index) + ".mat"))
    A_image = np.expand_dims(feature_dict['a'], axis=2)
    # print("A:", A_image.shape)

    ele_image = np.expand_dims(feature_dict['ele'], axis=2)
    # print("ele:", ele_image.shape)

    L_image = np.expand_dims(feature_dict['l'], axis=2)
    # print("L:", L_image.shape)

    azi_image = np.expand_dims(feature_dict['azi'], axis=2)
    # print("azi:", azi_image.shape)

    B_image = np.expand_dims(feature_dict['b'], axis=2)
    # print("B:", B_image.shape)

    entpy_image = np.expand_dims(feature_dict['entpy'], axis=2)
    # print("entpy:", entpy_image.shape)

    sat_image = np.expand_dims(feature_dict['sat'], axis=2)
    # print("sat:", sat_image.shape)

    entpy2_image = np.expand_dims(feature_dict['entpy2'], axis=2)
    # print('entpy2:', entpy2_image.shape)

    ndvi_image = np.expand_dims(feature_dict['ndvi'], axis=2)
    # print('ndvi:', ndvi_image.shape)

    return np.concatenate((top_image, ndsm_image, dsm_image, A_image, azi_image, B_image, ele_image,
                           entpy_image, entpy2_image, L_image, ndvi_image, sat_image, texton_image), axis=2).astype(dtype=np.float16)

for i in file_indices:
    np.save("../ISPRS_semantic_labeling_Vaihingen/top_15_channels/" + "top_mosaic_09cm_area" + str(i) + ".npy", get_15_channel_array(i))
    print("saved: ", i)
