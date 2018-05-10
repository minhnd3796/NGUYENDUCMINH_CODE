from cv2 import imread, imwrite
from os.path import join, splitext, exists
from os import listdir, mkdir
import numpy as np
from libtiff import TIFF

base_dir_top = "../ISPRS_semantic_labeling_Potsdam/4_Ortho_RGBIR"
base_dir_ndsm = "../ISPRS_semantic_labeling_Potsdam/1_DSM_normalisation"
base_dir_dsm = "../ISPRS_semantic_labeling_Potsdam/1_DSM"
base_dir_annotations = "../ISPRS_semantic_labeling_Potsdam/annotations"
save_dir = "../ISPRS_semantic_labeling_Potsdam/npy_6_channel"

def get_six_channel_image(filename):
    top_img = TIFF.open(join(base_dir_top, splitext(filename)[0].replace('label','RGBIR')+".tif"), 'r').read_image()
    element = filename.split('_')
    if len(element[2]) == 1:
        element[2] = '0' + element[2]
    if len(element[3]) == 1:
        element[3] = '0' + element[3]
    dsm_image_name= 'dsm' + '_' + element[1]+"_"+element[2]+"_"+element[3]+".tif"
    dsm_image = TIFF.open(join(base_dir_dsm, dsm_image_name), 'r').read_image()
    ndsm_image_name = dsm_image_name.replace('.tif', '') + "_normalized_ownapproach.jpg"
    ndsm_image = imread(join(base_dir_ndsm, ndsm_image_name), -1)
    return np.concatenate((top_img, np.expand_dims(ndsm_image, axis=2), np.expand_dims(dsm_image, axis=2)), axis=2)

if __name__ == '__main__':
    for filename in listdir(base_dir_annotations):
        print(filename)
        np.save(join(save_dir, splitext(filename)[0] + ".npy"), get_six_channel_image(filename))
