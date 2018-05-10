from cv2 import imread, imwrite
from os.path import join, splitext, exists
from os import listdir, mkdir
import numpy as np
from libtiff import TIFF

base_dir_top = "../ISPRS_semantic_labeling_Potsdam/4_Ortho_RGBIR"
base_dir_ndsm = "../ISPRS_semantic_labeling_Potsdam/1_DSM_normalisation"
base_dir_dsm = "../ISPRS_semantic_labeling_Potsdam/1_DSM"
save_dir = "../ISPRS_semantic_labeling_Potsdam/npy_6_channel"

def get_six_channel_image(filename):
    # top_img = TIFF.open(join(base_dir_top, splitext(filename)[0].replace('label','RGBIR')+".tif"), 'r').read_image()
    top_img = TIFF.open(join(base_dir_top, filename), 'r').read_image()
    
    element = filename.split('_')
    if len(element[2]) == 1:
        element[2] = '0' + element[2]
    if len(element[3]) == 1:
        element[3] = '0' + element[3]
    dsm_image_name= 'dsm' + '_' + element[1]+"_"+element[2]+"_"+element[3]+".tif"
    dsm_image = TIFF.open(join(base_dir_dsm, dsm_image_name), 'r').read_image()
    ndsm_image_name = dsm_image_name.replace('.tif', '') + "_normalized_lastools.jpg"
    ndsm_image = imread(join(base_dir_ndsm, ndsm_image_name), -1)
    print(top_img.shape, np.expand_dims(ndsm_image, axis=2).shape, np.expand_dims(dsm_image, axis=2).shape)
    if filename != "top_potsdam_3_13_RGBIR.tif":
        return np.concatenate((top_img, np.expand_dims(ndsm_image, axis=2), np.expand_dims(dsm_image, axis=2)), axis=2).astype(dtype=np.float16)
    else:
        dsm_image_2 = np.zeros((6000, 6000))
        ndsm_image_2 = np.zeros((6000, 6000))
        dsm_image_2[:, :5999] += dsm_image
        ndsm_image_2[:, :5999] += ndsm_image
        dsm_image_2[:, 5999] = dsm_image[:, 5998]
        ndsm_image_2[:, 5999] = ndsm_image[:, 5998]
        return np.concatenate((top_img, np.expand_dims(ndsm_image_2, axis=2), np.expand_dims(dsm_image_2, axis=2)), axis=2).astype(dtype=np.float16)

if __name__ == '__main__':
    """ for filename in listdir(base_dir_top):
        print(">> Processing:", filename)
        np.save(join(save_dir, splitext(filename)[0] + ".npy"), get_six_channel_image(filename)) """
    np.save("top_potsdam_3_13_RGBIR.npy", get_six_channel_image("top_potsdam_3_13_RGBIR.tif"))
