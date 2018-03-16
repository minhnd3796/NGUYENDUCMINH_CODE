import csv
import os

import numpy as np
from scipy.misc import imread


def convert2csv(image_dir):
    for filename in os.listdir(image_dir):
        image = imread(image_dir+"/"+filename)
        print(filename)
        height= np.shape(image)[0]
        width= np.shape(image)[1]
        print(height,width)
        pixel_pred=[]
        for i in range(width):
            for j in range(height):
                if np.array_equal(image[j,i],[255,255,255]):
                    pixel_pred.append(0)
                elif np.array_equal(image[j,i],[0,0,255]):
                    pixel_pred.append(1)
                elif np.array_equal(image[j,i],[0,255,255]):
                    pixel_pred.append(2)
                elif np.array_equal(image[j,i],[0,255,0]):
                    pixel_pred.append(3)
                elif np.array_equal(image[j,i],[255,255,0]):
                    pixel_pred.append(4)
                elif np.array_equal(image[j,i],[255,0,0]):
                    pixel_pred.append(5)
        print(len(pixel_pred))
        with open('image2csv/'+filename.split(".")[0]+'.csv','w') as csv_file:
            wr= csv.writer(csv_file)
            wr.writerow(['x'])
            for pred in pixel_pred:
                wr.writerow([pred])
        print('Done for '+ filename)
        pixel_pred=[]

if __name__=='__main__':
    convert2csv('infer_test')




# image = imread('top_mosaic_09cm_area7.tif')
# height = np.shape(image)[0]
# width = np.shape(image)[1]
# print(height, width)
# pixel_pred = []
# for i in range(width):
#     for j in range(height):
#         if np.array_equal(image[j,i], [255, 255, 255]):
#             pixel_pred.append(0)
#         elif np.array_equal(image[j,i], [0, 0, 255]):
#             pixel_pred.append(1)
#         elif np.array_equal(image[j,i], [0, 255, 255]):
#             pixel_pred.append(2)
#         elif np.array_equal(image[j,i], [0, 255, 0]):
#             pixel_pred.append(3)
#         elif np.array_equal(image[j,i], [255, 255, 0]):
#             pixel_pred.append(4)
#         elif np.array_equal(image[j,i], [255, 0, 0]):
#             pixel_pred.append(5)
# print(len(pixel_pred))
# print(pixel_pred[0])
# print(pixel_pred[1995])
# print(pixel_pred[1996])
# with open('top_mosaic_09cm_area7.csv', 'w') as csv_file:
#     wr = csv.writer(csv_file)
#     wr.writerow(['x'])
#     for pred in pixel_pred:
#         wr.writerow([pred])
# print('Done for ' + 'top_mosaic_09cm_area7.tif')
