import os

import numpy as np
import scipy.misc as misc

base_dir_train = "../ISPRS_semantic_labeling_Vaihingen/train_15channels"
base_dir_validate = "../ISPRS_semantic_labeling_Vaihingen/validate_15channels"
base_dir_annotations = "../ISPRS_semantic_labeling_Vaihingen/annotations"
base_dir_top = "../ISPRS_semantic_labeling_Vaihingen/top"
base_dir_ndsm = "../ISPRS_semantic_labeling_Vaihingen/ndsm"
base_dir_dsm = "../ISPRS_semantic_labeling_Vaihingen/dsm"
base_dir_ndvi= "../ISPRS_semantic_labeling_Vaihingen/ndvi"
base_dir_L= "../ISPRS_semantic_labeling_Vaihingen/L"
base_dir_A= "../ISPRS_semantic_labeling_Vaihingen/A"
base_dir_B= "../ISPRS_semantic_labeling_Vaihingen/B"
base_dir_ele= "../ISPRS_semantic_labeling_Vaihingen/ele"
base_dir_azi= "../ISPRS_semantic_labeling_Vaihingen/azi"
base_dir_sat= "../ISPRS_semantic_labeling_Vaihingen/sat"
base_dir_entpy= "../ISPRS_semantic_labeling_Vaihingen/entpy"
base_dir_entpy2= "../ISPRS_semantic_labeling_Vaihingen/entpy2"
base_dir_texton= "../ISPRS_semantic_labeling_Vaihingen/texton"
base_dir_train_validate_gt = "../ISPRS_semantic_labeling_Vaihingen/train_validate_gt_15channels"
image_size = 224
num_cropping_per_image = 3333
validate_image=['top_mosaic_09cm_area11.png']

def create_training_dataset():
    for filename in os.listdir(base_dir_annotations):
        if filename in validate_image:
            continue
        top_image = misc.imread(os.path.join(base_dir_top,os.path.splitext(filename)[0]+".tif"))
        annotation_image = misc.imread(os.path.join(base_dir_annotations, filename))
        dsm_image_name= filename.replace('top_mosaic','dsm').replace('png','tif').replace('area','matching_area')
        dsm_image= misc.imread(base_dir_dsm+"/"+dsm_image_name)
        ndsm_image_name= dsm_image_name.replace('.tif','')+"_normalized.jpg"
        ndsm_image= misc.imread(base_dir_ndsm+"/"+ndsm_image_name)
        A_image_name = "A"+ndsm_image_name.replace('dsm_09cm_matching_area','').replace('_normalized.jpg','.tif')
        A_image = misc.imread(base_dir_A + "/"+ A_image_name)
        azi_image_name = A_image_name.replace('A','azi')
        azi_image = misc.imread(base_dir_azi+"/"+azi_image_name)
        B_image_name = A_image_name.replace('A', 'B')
        B_image = misc.imread(base_dir_B + "/" + B_image_name)
        ele_image_name = A_image_name.replace('A', 'ele')
        ele_image = misc.imread(base_dir_ele + "/" + ele_image_name)
        entpy_image_name = A_image_name.replace('A', 'entpy')
        entpy_image = misc.imread(base_dir_entpy + "/" + entpy_image_name)
        entpy2_image_name = A_image_name.replace('A', 'entpy2')
        entpy2_image = misc.imread(base_dir_entpy2 + "/" + entpy2_image_name)
        L_image_name = A_image_name.replace('A', 'L')
        L_image = misc.imread(base_dir_L + "/" + L_image_name)
        ndvi_image_name = A_image_name.replace('A', 'ndvi')
        ndvi_image = misc.imread(base_dir_ndvi + "/" + ndvi_image_name)
        sat_image_name = A_image_name.replace('A', 'sat')
        sat_image = misc.imread(base_dir_sat + "/" + sat_image_name)
        texton_image_name = A_image_name.replace('A', 'texton')
        texton_image = misc.imread(base_dir_texton + "/" + texton_image_name)

        width= np.shape(top_image)[1]
        height= np.shape(top_image)[0]
        for i in range(num_cropping_per_image):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x,y))
            top_image_cropped= top_image[x:x + image_size, y:y + image_size, :]
            ndsm_image_cropped= ndsm_image[x:x + image_size, y:y + image_size]
            ndsm_image_cropped= np.expand_dims(ndsm_image_cropped,axis=2)
            dsm_image_cropped= dsm_image[x:x + image_size, y:y + image_size]
            dsm_image_cropped= np.expand_dims(dsm_image_cropped,axis=2)
            A_image_cropped = A_image[x:x + image_size, y:y + image_size]
            A_image_cropped = np.expand_dims(A_image_cropped, axis=2)
            azi_image_cropped = azi_image[x:x + image_size, y:y + image_size]
            azi_image_cropped = np.expand_dims(azi_image_cropped, axis=2)
            B_image_cropped = B_image[x:x + image_size, y:y + image_size]
            B_image_cropped = np.expand_dims(B_image_cropped, axis=2)
            ele_image_cropped = ele_image[x:x + image_size, y:y + image_size]
            ele_image_cropped = np.expand_dims(ele_image_cropped, axis=2)
            entpy_image_cropped = entpy_image[x:x + image_size, y:y + image_size]
            entpy_image_cropped = np.expand_dims(entpy_image_cropped, axis=2)
            entpy2_image_cropped = entpy2_image[x:x + image_size, y:y + image_size]
            entpy2_image_cropped = np.expand_dims(entpy2_image_cropped, axis=2)
            L_image_cropped = L_image[x:x + image_size, y:y + image_size]
            L_image_cropped = np.expand_dims(L_image_cropped, axis=2)
            ndvi_image_cropped = ndvi_image[x:x + image_size, y:y + image_size]
            ndvi_image_cropped = np.expand_dims(ndvi_image_cropped, axis=2)
            sat_image_cropped = sat_image[x:x + image_size, y:y + image_size]
            sat_image_cropped = np.expand_dims(sat_image_cropped, axis=2)
            texton_image_cropped = texton_image[x:x + image_size, y:y + image_size]
            texton_image_cropped = np.expand_dims(texton_image_cropped, axis=2)
            array_for_save= np.concatenate((top_image_cropped,ndsm_image_cropped,dsm_image_cropped, A_image_cropped,
                                            azi_image_cropped, B_image_cropped, ele_image_cropped, entpy_image_cropped, entpy2_image_cropped,
                                            L_image_cropped, ndvi_image_cropped, sat_image_cropped, texton_image_cropped),axis=2).astype(dtype=np.float16)
            np.save(os.path.join(base_dir_train, os.path.splitext(filename)[0] + "_" + str(i)+".npy"),array_for_save)
            annotation_image_cropped= annotation_image[x:x + image_size, y:y + image_size]
            misc.imsave(os.path.join(base_dir_train_validate_gt, os.path.splitext(filename)[0] + "_" + str(i) + ".png"), annotation_image_cropped)
    return None


def create_validation_dataset():
    for filename in validate_image:
        top_image = misc.imread(os.path.join(base_dir_top, os.path.splitext(filename)[0] + ".tif"))
        annotation_image = misc.imread(os.path.join(base_dir_annotations, filename))
        dsm_image_name = filename.replace('top_mosaic', 'dsm').replace('png', 'tif').replace('area','matching_area')
        dsm_image = misc.imread(base_dir_dsm + "/" + dsm_image_name)
        ndsm_image_name = dsm_image_name.replace('.tif', '') + "_normalized.jpg"
        ndsm_image = misc.imread(base_dir_ndsm + "/" + ndsm_image_name)
        width = np.shape(top_image)[1]
        height = np.shape(top_image)[0]
        for i in range(num_cropping_per_image):
            x = int(np.random.uniform(0, height - image_size + 1))
            y = int(np.random.uniform(0, width - image_size + 1))
            print((x, y))
            top_image_cropped = top_image[x:x + image_size, y:y + image_size, :]
            ndsm_image_cropped = ndsm_image[x:x + image_size, y:y + image_size]
            ndsm_image_cropped = np.expand_dims(ndsm_image_cropped, axis=2)
            dsm_image_cropped = dsm_image[x:x + image_size, y:y + image_size]
            dsm_image_cropped = np.expand_dims(dsm_image_cropped, axis=2)
            array_for_save = np.concatenate((top_image_cropped, ndsm_image_cropped, dsm_image_cropped), axis=2).astype(dtype=np.float16)
            np.save(os.path.join(base_dir_validate, os.path.splitext(filename)[0] + "_" + str(i) + ".npy"), array_for_save)
            # misc.imsave(os.path.join(base_dir_train, os.path.splitext(filename)[0] + "_" + str(i) + ".tif"), top_image_cropped)
            annotation_image_cropped = annotation_image[x:x + image_size, y:y + image_size]
            misc.imsave(os.path.join(base_dir_train_validate_gt, os.path.splitext(filename)[0] + "_" + str(i) + ".png"),
                        annotation_image_cropped)
    return None

if __name__=="__main__":
    create_training_dataset()
