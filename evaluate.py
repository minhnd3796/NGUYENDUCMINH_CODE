import numpy as np
# from scipy.misc import imread
from cv2 import imread
from sys import argv
from os import listdir
from os.path import join

def evaluate_dir(log_dir, epoch_num, acc_logfile, train_or_val_dir):
    match = 0
    num_pix = 0
    epoch_num_dir = 'epoch_' + epoch_num

    for image in listdir(join(log_dir, 'inferred_images', epoch_num_dir, train_or_val_dir)):
        pred = imread(join(log_dir, 'inferred_images', epoch_num_dir, train_or_val_dir, image))
        annotation = imread(join('../ISPRS_semantic_labeling_Vaihingen/gts_for_participants/', image))
        height = np.shape(pred)[0]
        width = np.shape(annotation)[1]
        num_pix += height * width
        print('>> Counting', image)
        for i in range(height):
            for j in range(width):
                if np.array_equal(pred[i,j,:], annotation[i,j,:]):
                    match += 1
    with open(join(log_dir, acc_logfile), 'a') as f:
        f.write(epoch_num + ',' + str(match / num_pix) + '\n')

if __name__ == '__main__':
    if argv[2] == 'val':
        validation_image = ['top_mosaic_09cm_area7', 'top_mosaic_09cm_area17',
                            'top_mosaic_09cm_area23', 'top_mosaic_09cm_area37']
    elif argv[2] == 'train':
        validation_image = ['top_mosaic_09cm_area1', 'top_mosaic_09cm_area3', 'top_mosaic_09cm_area5',
                            'top_mosaic_09cm_area11', 'top_mosaic_09cm_area13', 'top_mosaic_09cm_area15',
                            'top_mosaic_09cm_area21', 'top_mosaic_09cm_area26', 'top_mosaic_09cm_area28',
                            'top_mosaic_09cm_area30', 'top_mosaic_09cm_area32', 'top_mosaic_09cm_area34']

    match = 0
    num_pix = 0

    inferred_dir = 'inferred_images/'

    for image in validation_image:
        pred = imread(inferred_dir + image + '_' + argv[1] + '.tif')
        annotation = imread('../ISPRS_semantic_labeling_Vaihingen/gts_for_participants/' + image + '.tif')
        height = np.shape(pred)[0]
        width = np.shape(annotation)[1]
        num_pix += height * width
        print('>> Evaluating', image + '_' + argv[1] + '.tif')
        for i in range(height):
            for j in range(width):
                if np.array_equal(pred[i,j,:],annotation[i,j,:]):
                    match += 1
    print("!! Accuracy:", match / num_pix)
# 7  Accuracy: 0.8903082843132074
# 17 Accuracy: 0.8781703479730091
# 23 Accuracy: 0.8522438833297077
# 37 Accuracy: 0.868925821567948

# 7_Accuracy: 0.8863893685030587
# 17_Accuracy: 0.8812551463432892
# 23_Accuracy: 0.8460748914662796
# 37_Accuracy: 0.8716171691754436

# 37_Accuracy: 0.8807770930331842
# 23_Accuracy: 0.8600766392337893
# 17 Accuracy: 0.8761184942200549
# 7 Accuracy: 0.8918958296675751


# 7 Accuracy: 0.8910807101011614
# 17 Accuracy: 0.8760065445446088
# 23 Accuracy: 0.8588578665430487
# 37 Accuracy: 0.8798381223600082
