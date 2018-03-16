import os

import numpy as np
import pandas as pd
from scipy.misc import imread, imsave

save_dir='/home/baduy9x/PycharmProjects/Themis/ISPRS_semantic_labeling_Vaihingen'
csv_dir='/home/baduy9x/Desktop/csv'
base_dir='/home/baduy9x/PycharmProjects/Themis/ISPRS_semantic_labeling_Vaihingen/top'
for filename in os.listdir(base_dir):
    image = imread(base_dir+"/"+filename)
    height = np.shape(image)[0]
    width = np.shape(image)[1]
    print(height,width)
    filename=filename.replace('.tif','\t').replace('top_mosaic_09cm_area','\t')
    image_number=filename.split('\t')
    df = pd.read_csv(csv_dir +"/feat" + image_number[1] + ".csv")
    df_texton = pd.read_csv(csv_dir +"/texton" + image_number[1] + ".csv")
    ndvi = df['ndvi'].tolist()
    temp_image = np.transpose(np.array(ndvi, dtype=np.float16).reshape((width,height)))
    del ndvi
    imsave(save_dir +'/ndvi/ndvi' + image_number[1] + ".tif", temp_image)
    sat = df['sat'].tolist()
    temp_image = np.transpose(np.array(sat, dtype=np.float16).reshape((width,height)))
    del sat
    imsave(save_dir + '/sat/sat' + image_number[1] + ".tif", temp_image)
    l = df['l'].tolist()
    temp_image = np.transpose(np.array(l, dtype=np.float16).reshape((width,height)))
    del l
    imsave(save_dir + '/L/L' + image_number[1] + ".tif", temp_image)
    a = df['a'].tolist()
    temp_image = np.transpose(np.array(a, dtype=np.float16).reshape((width,height)))
    del a
    imsave(save_dir + '/A/A' + image_number[1] + ".tif", temp_image)
    b = df['b'].tolist()
    temp_image = np.transpose(np.array(b, dtype=np.float16).reshape((width,height)))
    del b
    imsave(save_dir + '/B/B' + image_number[1] + ".tif", temp_image)
    azi = df['azi'].tolist()
    temp_image = np.transpose(np.array(azi, dtype=np.float16).reshape((width,height)))
    del azi
    imsave(save_dir + '/azi/azi' + image_number[1] + ".tif", temp_image)
    ele = df['ele'].tolist()
    temp_image = np.transpose(np.array(ele, dtype=np.float16).reshape((width,height)))
    del ele
    imsave(save_dir + '/ele/ele' + image_number[1] + ".tif", temp_image)
    entpy = df['entpy'].tolist()
    temp_image = np.transpose(np.array(entpy, dtype=np.float16).reshape((width,height)))
    del entpy
    imsave(save_dir + '/entpy/entpy' + image_number[1] + ".tif", temp_image)
    entpy2= df['entpy2'].tolist()
    temp_image = np.transpose(np.array(entpy2, dtype=np.float16).reshape((width,height)))
    del entpy2
    imsave(save_dir + '/entpy2/entpy2' + image_number[1] + ".tif", temp_image)
    texton = df_texton['texton'].tolist()
    temp_image = np.transpose(np.array(texton, dtype=np.float16).reshape((width,height)))
    del texton
    imsave(save_dir + '/texton/texton' + image_number[1] + ".tif", temp_image)
