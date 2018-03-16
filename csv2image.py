import csv
import os
import re

import numpy as np
from scipy.misc import imread, imsave


def csv2img(csv_dir):
    for filename in os.listdir(csv_dir):
        with open(csv_dir + "/" + filename) as csv_file:
            readCSV = csv.reader(csv_file)
            j = 0
            image_number = re.findall(r'\d+', filename)
            print(filename)
            image = imread("infer_test/top_mosaic_09cm_area" + image_number[0] + ".tif")
            output_image = np.zeros_like(image)
            height = np.shape(image)[0]
            width = np.shape(image)[1]
            pred_list = []
            for row in readCSV:
                if j == 0:
                    j += 1
                    continue
                pred_list.append(row[0])
            pred_matrix = np.array(pred_list)
            pred_matrix = np.reshape(pred_matrix, (width, height)).transpose()
            for m in range(height):
                for n in range(width):
                    if int(pred_matrix[m, n]) == 0:
                        output_image[m, n] = np.array([255, 255, 255])
                    elif int(pred_matrix[m, n]) == 1:
                        output_image[m, n] = np.array([0, 0, 255])
                    elif int(pred_matrix[m, n]) == 2:
                        output_image[m, n] = np.array([0, 255, 255])
                    elif int(pred_matrix[m, n]) == 3:
                        output_image[m, n] = np.array([0, 255, 0])
                    elif int(pred_matrix[m, n]) == 4:
                        output_image[m, n] = np.array([255, 255, 0])
                    elif int(pred_matrix[m, n]) == 5:
                        output_image[m, n] = np.array([255, 0, 0])
            imsave('crf_infer_test/top_mosaic_09cm_area' + image_number[0] + "_class.tif", output_image)


if __name__ == '__main__':
    csv2img('crf-output')
