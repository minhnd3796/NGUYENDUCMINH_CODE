from cv2 import imread, imwrite
import numpy as np
from os.path import join
from sys import argv

def create_patch_batch_list(filename,
                            batch_size,
                            data_dir=join('..', 'ISPRS_semantic_labeling_Vaihingen'),
                            num_channels=3,
                            patch_size=224,
                            vertical_stride=112,
                            horizontal_stride=112):
    input_batch_list = []
    gt_annotation_batch_list = []
    coordinate_batch_list = []
    top_dir = join(data_dir, 'top')
    global_patch_index = 0
    
    top_img = imread(join(top_dir, filename + '.tif'))
    height = top_img.shape[0]
    width = top_img.shape[1]
    num_vertical_points = (height - patch_size) // vertical_stride + 1
    num_horizontial_points = (width - patch_size) // horizontal_stride + 1
    if num_channels == 3:
        input_img = top_img
    elif num_channels == 5:
        ndsm_dir = join(data_dir, 'ndsm')
        ndsm_img = imread(join(ndsm_dir, filename.replace('top', 'dsm').replace('_mosaic', '').replace('area', 'matching_area') + '_normalized.jpg'), -1)
        ndsm_img = ndsm_img[:, :, np.newaxis]
        dsm_dir = join(data_dir, 'dsm')
        dsm_img = imread(join(dsm_dir, filename.replace('top', 'dsm').replace('_mosaic', '').replace('area', 'matching_area') + '.tif'), -1)
        dsm_img = dsm_img[:, :, np.newaxis]
        input_img = np.concatenate((top_img, ndsm_img, dsm_img), axis=2)
    for i in range(num_vertical_points):
        for j in range(num_horizontial_points):
            local_patch_index = global_patch_index % batch_size
            if local_patch_index == 0:
                current_input_batch = np.zeros((batch_size, patch_size, patch_size, num_channels))
                current_gt_annotation_batch = np.array([None] * batch_size)
                current_coordinate_batch = [(None, None)] * batch_size
            current_coordinate_batch[local_patch_index] = (i * vertical_stride, j * horizontal_stride)
            current_input_batch[local_patch_index, :, :, :] = input_img[current_coordinate_batch[local_patch_index][0]:current_coordinate_batch[local_patch_index][0] + patch_size,
                                                               current_coordinate_batch[local_patch_index][1]:current_coordinate_batch[local_patch_index][1] + patch_size,:]
            if local_patch_index == batch_size - 1:
                input_batch_list.append(current_input_batch)
                gt_annotation_batch_list.append(current_gt_annotation_batch)
                coordinate_batch_list.append(current_coordinate_batch)
            global_patch_index += 1
    for i in range(num_vertical_points):
        local_patch_index = global_patch_index % batch_size
        if local_patch_index == 0:
            current_input_batch = np.zeros((batch_size, patch_size, patch_size, num_channels))
            current_gt_annotation_batch = np.array([None] * batch_size)
            current_coordinate_batch = [(None, None)] * batch_size
        current_coordinate_batch[local_patch_index] = (i * vertical_stride, width - patch_size)
        current_input_batch[local_patch_index, :, :, :] = input_img[current_coordinate_batch[local_patch_index][0]:current_coordinate_batch[local_patch_index][0] + patch_size,
                                                           current_coordinate_batch[local_patch_index][1]:current_coordinate_batch[local_patch_index][1] + patch_size,:]
        if local_patch_index == batch_size - 1:
            input_batch_list.append(current_input_batch)
            gt_annotation_batch_list.append(current_gt_annotation_batch)
            coordinate_batch_list.append(current_coordinate_batch)
        global_patch_index += 1
    for i in range(num_horizontial_points):
        local_patch_index = global_patch_index % batch_size
        if local_patch_index == 0:
            current_input_batch = np.zeros((batch_size, patch_size, patch_size, num_channels))
            current_gt_annotation_batch = np.array([None] * batch_size)
            current_coordinate_batch = [(None, None)] * batch_size
        current_coordinate_batch[local_patch_index] = (height - patch_size, i * horizontal_stride)
        current_input_batch[local_patch_index, :, :, :] = input_img[current_coordinate_batch[local_patch_index][0]:current_coordinate_batch[local_patch_index][0] + patch_size,
                                                           current_coordinate_batch[local_patch_index][1]:current_coordinate_batch[local_patch_index][1] + patch_size,:]
        if local_patch_index == batch_size - 1:
            input_batch_list.append(current_input_batch)
            gt_annotation_batch_list.append(current_gt_annotation_batch)
            coordinate_batch_list.append(current_coordinate_batch)
        global_patch_index += 1
    local_patch_index = global_patch_index % batch_size
    if local_patch_index == 0:
        current_input_batch = np.zeros((batch_size, patch_size, patch_size, num_channels))
        current_gt_annotation_batch = np.array([None] * batch_size)
        current_coordinate_batch = [(None, None)] * batch_size
    current_coordinate_batch[local_patch_index] = (height - patch_size, width - patch_size)
    current_input_batch[local_patch_index, :, :, :] = input_img[current_coordinate_batch[local_patch_index][0]:current_coordinate_batch[local_patch_index][0] + patch_size,
                                                       current_coordinate_batch[local_patch_index][1]:current_coordinate_batch[local_patch_index][1] + patch_size,:]
    if local_patch_index == batch_size - 1:
        input_batch_list.append(current_input_batch)
        gt_annotation_batch_list.append(current_gt_annotation_batch)
        coordinate_batch_list.append(current_coordinate_batch)
    global_patch_index += 1
    local_patch_index = global_patch_index % batch_size
    if local_patch_index != 0:
        for i in range(batch_size - local_patch_index):
            local_patch_index = global_patch_index % batch_size
            current_coordinate_batch[local_patch_index] = (height - patch_size, width - patch_size)
            current_input_batch[local_patch_index, :, :, :] = input_img[current_coordinate_batch[local_patch_index][0]:current_coordinate_batch[local_patch_index][0] + patch_size,
                                                               current_coordinate_batch[local_patch_index][1]:current_coordinate_batch[local_patch_index][1] + patch_size,:]
            if local_patch_index == batch_size - 1:
                input_batch_list.append(current_input_batch)
                gt_annotation_batch_list.append(current_gt_annotation_batch)
                coordinate_batch_list.append(current_coordinate_batch)
            global_patch_index += 1
    return input_batch_list, coordinate_batch_list, height, width

def batch_inference(input_tensor,
                    logits,
                    keep_probability,
                    encoding_keep_prob,
                    sess,
                    input_batch_list,
                    coordinate_batch_list,
                    height,
                    width,
                    patch_size=224):
    logits_map = np.zeros((height, width, 6), dtype=np.float32)
    for i in range(len(input_batch_list)):
        if encoding_keep_prob == None:
            logits_batch = sess.run(logits, feed_dict={input_tensor: input_batch_list[i], keep_probability: 1.0})
        else:
            logits_batch = sess.run(logits, feed_dict={input_tensor: input_batch_list[i], keep_probability: 1.0, encoding_keep_prob: 1.0})
        for j in range(logits_batch.shape[0]):
            logits_map[coordinate_batch_list[i][j][0]:coordinate_batch_list[i][j][0] + patch_size,
                       coordinate_batch_list[i][j][1]:coordinate_batch_list[i][j][1] + patch_size, :] += logits_batch[j]
    return np.argmax(logits_map, axis=2)

def eval_dir(input_tensor,
             logits,
             keep_probability,
             sess,
             batch_size,
             log_dir,
             epoch_num,
             encoding_keep_prob=None,
             num_channels=3,
             patch_size=224,
             vertical_stride=112,
             horizontal_stride=112,
             is_validation=True):
    if is_validation:
        # filename = ['top_mosaic_09cm_area7', 'top_mosaic_09cm_area17', 'top_mosaic_09cm_area23', 'top_mosaic_09cm_area37']
        filename = ['top_mosaic_09cm_area7']
        acc_logfile = 'epoch_val_acc.csv'
    else:
        filename = ['top_mosaic_09cm_area1', 'top_mosaic_09cm_area3']
        """ filename = ['top_mosaic_09cm_area1', 'top_mosaic_09cm_area3', 'top_mosaic_09cm_area5',
                    'top_mosaic_09cm_area11', 'top_mosaic_09cm_area13', 'top_mosaic_09cm_area15',
                    'top_mosaic_09cm_area21', 'top_mosaic_09cm_area26', 'top_mosaic_09cm_area28',
                    'top_mosaic_09cm_area30', 'top_mosaic_09cm_area32', 'top_mosaic_09cm_area34'] """

        # For submission only
        """ filename = ['top_mosaic_09cm_area1', 'top_mosaic_09cm_area3', 'top_mosaic_09cm_area5',
                    'top_mosaic_09cm_area11', 'top_mosaic_09cm_area13', 'top_mosaic_09cm_area15',
                    'top_mosaic_09cm_area21', 'top_mosaic_09cm_area26', 'top_mosaic_09cm_area28',
                    'top_mosaic_09cm_area30', 'top_mosaic_09cm_area32', 'top_mosaic_09cm_area34',
                    'top_mosaic_09cm_area7', 'top_mosaic_09cm_area17', 'top_mosaic_09cm_area23',
                    'top_mosaic_09cm_area37'] """
        acc_logfile = 'epoch_train_acc.csv'
    num_matches = 0
    num_pixels = 0
    for fn in filename:
        input_batch_list, coordinate_batch_list, height, width = create_patch_batch_list(fn, batch_size, num_channels=num_channels)
        pred_annotation_map = batch_inference(input_tensor, logits, keep_probability, encoding_keep_prob, sess, input_batch_list, coordinate_batch_list, height, width)
        num_matches += np.sum(pred_annotation_map == imread("../ISPRS_semantic_labeling_Vaihingen/annotations/" + fn + ".png", -1))
        num_pixels += pred_annotation_map.shape[0] * pred_annotation_map.shape[1]
    with open(join(log_dir, acc_logfile), 'a') as f:
        f.write(str(epoch_num) + ',' + str(num_matches / num_pixels) + '\n')

def get_patches(image_name, patch_size=224, vertical_stride=112, horizontal_stride=112):
    input_img = imread("../ISPRS_semantic_labeling_Vaihingen/top/" + image_name + ".tif")
    annotation_img = imread("../ISPRS_semantic_labeling_Vaihingen/annotations/" + image_name + '.png', -1)
    height = np.shape(input_img)[0]
    width = np.shape(input_img)[1]
    number_of_vertical_points = (height - patch_size) // vertical_stride + 1
    number_of_horizontial_points = (width - patch_size) // horizontal_stride + 1
    input_patch = []
    gt_patch = []
    coordinate = []
    num_patches = 0
    for i in range(number_of_vertical_points):
        for j in range(number_of_horizontial_points):
            current_coodinate = (i * vertical_stride, j * horizontal_stride)
            current_input_patch = input_img[current_coodinate[0]:current_coodinate[0] + patch_size,
                                            current_coodinate[1]:current_coodinate[1] + patch_size,:]
            current_gt_patch = annotation_img[current_coodinate[0]:current_coodinate[0] + patch_size,
                                              current_coodinate[1]:current_coodinate[1] + patch_size]
            coordinate.append(current_coodinate)
            gt_patch.append(current_gt_patch)
            input_patch.append(current_input_patch)
            num_patches += 1
    for i in range(number_of_vertical_points):
        current_coodinate = (i * vertical_stride, width - patch_size)
        current_input_patch = input_img[current_coodinate[0]:current_coodinate[0] + patch_size,
                                        current_coodinate[1]:current_coodinate[1] + patch_size,:]
        current_gt_patch = annotation_img[current_coodinate[0]:current_coodinate[0] + patch_size,
                                          current_coodinate[1]:current_coodinate[1] + patch_size]
        coordinate.append(current_coodinate) 
        gt_patch.append(current_gt_patch)
        input_patch.append(current_input_patch)
        num_patches += 1
    for i in range(number_of_horizontial_points):
        current_coodinate = (height - patch_size, i * horizontal_stride)
        current_input_patch = input_img[current_coodinate[0]:current_coodinate[0] + patch_size,
                                        current_coodinate[1]:current_coodinate[1] + patch_size,:]
        current_gt_patch = annotation_img[current_coodinate[0]:current_coodinate[0] + patch_size,
                                          current_coodinate[1]:current_coodinate[1] + patch_size]
        coordinate.append(current_coodinate)
        gt_patch.append(current_gt_patch)
        input_patch.append(current_input_patch)
        num_patches += 1
    current_coodinate = (height - patch_size, width - patch_size)
    current_input_patch = input_img[current_coodinate[0]:current_coodinate[0] + patch_size,
                                    current_coodinate[1]:current_coodinate[1] + patch_size,:]
    current_gt_patch = annotation_img[current_coodinate[0]:current_coodinate[0] + patch_size,
                                        current_coodinate[1]:current_coodinate[1] + patch_size]
    coordinate.append(current_coodinate)
    gt_patch.append(current_gt_patch)
    input_patch.append(current_input_patch)
    num_patches += 1
    return input_patch, gt_patch, coordinate, num_patches

if __name__ == '__main__':
    filename = ['top_mosaic_09cm_area1', 'top_mosaic_09cm_area3', 'top_mosaic_09cm_area5',
                'top_mosaic_09cm_area11', 'top_mosaic_09cm_area13', 'top_mosaic_09cm_area15',
                'top_mosaic_09cm_area21', 'top_mosaic_09cm_area26', 'top_mosaic_09cm_area28',
                'top_mosaic_09cm_area30', 'top_mosaic_09cm_area32', 'top_mosaic_09cm_area34',
                'top_mosaic_09cm_area7', 'top_mosaic_09cm_area17', 'top_mosaic_09cm_area23',
                'top_mosaic_09cm_area37']
    batch_size = int(argv[1])
    for fn in filename:
        input_batch_list, coordinate_batch_list, _, _ = \
            create_patch_batch_list(fn, batch_size)
        input_patch, gt_patch, coordinate, num_patches = get_patches(fn)

        global_i = 0
        input_diff = [None] * len(input_patch)
        y_diff = [None] * len(input_patch)
        x_diff = [None] * len(input_patch)
        

        for i in range(len(input_batch_list)):
            for j in range(batch_size):
                input_diff[global_i] = np.sum(input_patch[global_i] - input_batch_list[i][j])
                y_diff[global_i] = coordinate[global_i][0] - coordinate_batch_list[i][j][0]
                x_diff[global_i] = coordinate[global_i][1] - coordinate_batch_list[i][j][1]
                global_i += 1
                if global_i == len(input_patch):
                    global_i -= 1
                    break
        print(fn)
        print("Input diff:", sum(input_diff))
        print('x diff:', sum(x_diff))
        print('y diff:', sum(y_diff))

        input_diff = []
        gtt_diff = []
        x_diff = []
        y_diff = []
        last = j
        
        for j in range(last, batch_size):
            input_diff.append(np.sum(input_patch[global_i] - input_batch_list[i][j]))
            y_diff.append(coordinate[global_i][0] - coordinate_batch_list[i][j][0])
            x_diff.append(coordinate[global_i][1] - coordinate_batch_list[i][j][1])
        print("Last Input diff:", sum(input_diff))
        print('last x diff:', sum(x_diff))
        print('last y diff:', sum(y_diff))
        print()