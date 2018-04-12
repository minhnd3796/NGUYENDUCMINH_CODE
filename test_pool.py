from cv2 import imread, imwrite
import numpy as np
from os import listdir
from os.path import splitext, join
from progress import printProgressBar
import multiprocessing as mp

input_dir = 'gts_for_participants'
annotation_dir = 'annotations'
output_dir = 'annotations_back2_gts'

def process(item):
    global input_dir
    global annotation_dir
    global output_dir
    img_gt = imread(join(input_dir, item))
    img_annotation = np.zeros((img_gt.shape[0], img_gt.shape[1]))
    for i in range(img_gt.shape[0]):
        for j in range(img_gt.shape[1]):
            # printProgressBar(i * img_gt.shape[1] + j, img_gt.shape[0] * img_gt.shape[1],
            #                 prefix='Annotating: '+item, suffix='Complete', length=50)
            if np.array_equal(img_gt[i, j, :], np.array([255, 255, 255])):
                img_annotation[i, j] = 0
            elif np.array_equal(img_gt[i, j, :], np.array([0, 0, 255])):
                img_annotation[i, j] = 1
            elif np.array_equal(img_gt[i, j, :], np.array([0, 255, 255])):
                img_annotation[i, j] = 2
            elif np.array_equal(img_gt[i, j, :], np.array([0, 255, 0])):
                img_annotation[i, j] = 3
            elif np.array_equal(img_gt[i, j, :], np.array([255, 255, 0])):
                img_annotation[i, j] = 4
            elif np.array_equal(img_gt[i, j, :], np.array([255, 0, 0])):
                img_annotation[i, j] = 5
    imwrite(join(annotation_dir, splitext(item)[0] + '.png'), img_annotation)
    img_annotation = imread(join(annotation_dir, splitext(item)[0] + '.png'), -1)
    img_inferred = np.zeros((img_annotation.shape[0], img_annotation.shape[1], 3), dtype=np.uint8)
    for i in range(img_annotation.shape[0]):
        for j in range(img_annotation.shape[1]):
            # printProgressBar(i * img_gt.shape[1] + j, img_annotation.shape[0] * img_annotation.shape[1],
            #                 prefix='Inferring: '+item, suffix='Complete', length=50)
            if img_annotation[i, j] == 0:
                img_inferred[i, j, :] = [255, 255, 255]
            elif img_annotation[i, j] == 1:
                img_inferred[i, j, :] = [0, 0, 255]
            elif img_annotation[i, j] == 2:
                img_inferred[i, j, :] = [0, 255, 255]
            elif img_annotation[i, j] == 3:
                img_inferred[i, j, :] = [0, 255, 0]
            elif img_annotation[i, j] == 4:
                img_inferred[i, j, :] = [255, 255, 0]
            elif img_annotation[i, j] == 5:
                img_inferred[i, j, :] = [255, 0, 0]
    imwrite(join(output_dir, splitext(item)[0])+'.tif', img_inferred)

pool = mp.Pool(16)
pool.map(process, listdir(input_dir))

""" for item in listdir(input_dir):
    img_gt = imread(join(input_dir, item))
    img_annotation = np.zeros((img_gt.shape[0], img_gt.shape[1]))
    for i in range(img_gt.shape[0]):
        for j in range(img_gt.shape[1]):
            printProgressBar(i * img_gt.shape[1] + j, img_gt.shape[0] * img_gt.shape[1],
                            prefix='Annotating: '+item, suffix='Complete', length=50)
            if np.array_equal(img_gt[i, j, :], np.array([255, 255, 255])):
                img_annotation[i, j] = 0
            elif np.array_equal(img_gt[i, j, :], np.array([0, 0, 255])):
                img_annotation[i, j] = 1
            elif np.array_equal(img_gt[i, j, :], np.array([0, 255, 255])):
                img_annotation[i, j] = 2
            elif np.array_equal(img_gt[i, j, :], np.array([0, 255, 0])):
                img_annotation[i, j] = 3
            elif np.array_equal(img_gt[i, j, :], np.array([255, 255, 0])):
                img_annotation[i, j] = 4
            elif np.array_equal(img_gt[i, j, :], np.array([255, 0, 0])):
                img_annotation[i, j] = 5
    imwrite(join(annotation_dir, splitext(item)[0] + '.png'), img_annotation)
    img_annotation = imread(join(annotation_dir, splitext(item)[0] + '.png'), -1)
    img_inferred = np.zeros((img_annotation.shape[0], img_annotation.shape[1], 3), dtype=np.uint8)
    for i in range(img_annotation.shape[0]):
        for j in range(img_annotation.shape[1]):
            printProgressBar(i * img_gt.shape[1] + j, img_annotation.shape[0] * img_annotation.shape[1],
                            prefix='Inferring: '+item, suffix='Complete', length=50)
            if img_annotation[i, j] == 0:
                img_inferred[i, j, :] = [255, 255, 255]
            elif img_annotation[i, j] == 1:
                img_inferred[i, j, :] = [0, 0, 255]
            elif img_annotation[i, j] == 2:
                img_inferred[i, j, :] = [0, 255, 255]
            elif img_annotation[i, j] == 3:
                img_inferred[i, j, :] = [0, 255, 0]
            elif img_annotation[i, j] == 4:
                img_inferred[i, j, :] = [255, 255, 0]
            elif img_annotation[i, j] == 5:
                img_inferred[i, j, :] = [255, 0, 0]
    imwrite(join(output_dir, splitext(item)[0])+'.tif', img_inferred) """