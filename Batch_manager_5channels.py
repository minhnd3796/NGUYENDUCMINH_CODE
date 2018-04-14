import numpy as np
from cv2 import imread, imwrite
import tensorflow as tf
from os.path import exists, join
from os import mkdir
from evaluate import evaluate_dir

def infer_img(input_tensor, logits, keep_probability, sess, image_name,patch_size=224,stride_ver=112,stride_hor=112,log_dir='',epoch_num='',is_train=False):
    input_image_path = "../ISPRS_semantic_labeling_Vaihingen/top/" + image_name + ".tif"
    input_image = imread(input_image_path)
    dsm_image = imread(input_image_path.replace('top','dsm').replace('_mosaic','').replace('area','matching_area'), -1)
    ndsm_image = imread(input_image_path.replace('top/','ndsm/').replace('top','dsm').replace('_mosaic','')
                       .replace('area','matching_area').replace('.tif','_normalized.jpg'), -1)
    dsm_image = np.expand_dims(dsm_image, axis=2)
    ndsm_image = np.expand_dims(ndsm_image, axis=2)
    height = np.shape(input_image)[0]
    width = np.shape(input_image)[1]
    output_image = np.zeros_like(input_image)
    input_image= np.concatenate((input_image,ndsm_image,dsm_image),axis=2)
    output_map = np.zeros((height, width, 6), dtype=np.float32)
    number_of_vertical_points = (height - patch_size) // stride_ver + 1
    number_of_horizontial_points = (width - patch_size) // stride_hor + 1

    print(">>> Inferring", image_name)
    input_image = np.expand_dims(input_image,axis=0)
    print("++> Stage 1...")
    for i in range(number_of_vertical_points):
        for j in range(number_of_horizontial_points):
            current_patch = input_image[:,i * stride_ver:i * stride_ver + patch_size,
                            j * stride_hor:j * stride_hor + patch_size, :]
            logits_result = sess.run(logits, feed_dict={input_tensor: current_patch, keep_probability: 1.0})
            logits_result = tf.squeeze(logits_result)
            patch_result= sess.run(logits_result)
            output_map[i * stride_ver:i * stride_ver + patch_size, j * stride_hor:j * stride_hor + patch_size,:] += patch_result
            # print('stage 1: i='+str(i)+"; j="+str(j))
    print("++> Stage 2...")
    for i in range(number_of_vertical_points):
        current_patch= input_image[:,i*stride_ver:i*stride_ver+patch_size,width-patch_size:width,:]
        logits_result = sess.run(logits, feed_dict={input_tensor: current_patch, keep_probability: 1.0})
        logits_result = tf.squeeze(logits_result)
        patch_result = sess.run(logits_result)
        output_map[i*stride_ver:i*stride_ver+patch_size,width-patch_size:width,:]+=patch_result
        # print('stage 2: i=' + str(i) + "; j=" + str(j))
    print("++> Stage 3...")
    for i in range(number_of_horizontial_points):
        current_patch= input_image[:,height-patch_size:height,i*stride_hor:i*stride_hor+patch_size,:]
        logits_result = sess.run(logits, feed_dict={input_tensor: current_patch, keep_probability: 1.0})
        logits_result = tf.squeeze(logits_result)
        patch_result = sess.run(logits_result)
        output_map[height-patch_size:height,i*stride_hor:i*stride_hor+patch_size,:]+=patch_result
        # print('stage 3: i=' + str(i) + "; j=" + str(j))
    current_patch = input_image[:,height - patch_size:height, width - patch_size:width, :]
    logits_result = sess.run(logits, feed_dict={input_tensor: current_patch, keep_probability: 1.0})
    logits_result = tf.squeeze(logits_result)
    patch_result = sess.run(logits_result)
    output_map[height - patch_size:height, width - patch_size:width, :] += patch_result
    predict_annotation_image = np.argmax(output_map, axis=2)
    # print(np.shape(predict_annotation_image))
    for i in range(height):
        for j in range(width):
            if predict_annotation_image[i,j]==0:
                output_image[i,j,:]=[255,255,255]
            elif predict_annotation_image[i,j]==1:
                output_image[i,j,:]=[0,0,255]
            elif predict_annotation_image[i,j]==2:
                output_image[i,j,:]=[0,255,255]
            elif predict_annotation_image[i,j]==3:
                output_image[i,j,:]=[0,255,0]
            elif predict_annotation_image[i,j]==4:
                output_image[i,j,:]=[255,255,0]
            elif predict_annotation_image[i,j]==5:
                output_image[i,j,:]=[255,0,0]
    if not exists(join(log_dir, 'inferred_images')):
        mkdir(join(log_dir, 'inferred_images'))
    if not exists(join(log_dir, 'inferred_images', 'epoch_' + epoch_num)):
        mkdir(join(log_dir, 'inferred_images', 'epoch_' + epoch_num))
    if is_train:
        train_or_val_dir = 'train'
    else:
        train_or_val_dir = 'val'
    if not exists(join(log_dir, 'inferred_images', 'epoch_' + epoch_num, train_or_val_dir)):
        mkdir(join(log_dir, 'inferred_images', 'epoch_' + epoch_num, train_or_val_dir))
    imwrite(join(log_dir, 'inferred_images', 'epoch_' + epoch_num, train_or_val_dir, image_name + '.tif'), output_image)

def infer_and_validate_all_images(input_tensor, logits, keep_probability, sess, epoch_num, log_dir):
    # validation_image = ['top_mosaic_09cm_area7', 'top_mosaic_09cm_area17', 'top_mosaic_09cm_area23', 'top_mosaic_09cm_area37']
    # training_image = ['top_mosaic_09cm_area1', 'top_mosaic_09cm_area3', 'top_mosaic_09cm_area5',
    #                   'top_mosaic_09cm_area11', 'top_mosaic_09cm_area13', 'top_mosaic_09cm_area15',
    #                   'top_mosaic_09cm_area21', 'top_mosaic_09cm_area26', 'top_mosaic_09cm_area28',
    #                   'top_mosaic_09cm_area30', 'top_mosaic_09cm_area32', 'top_mosaic_09cm_area34']
    validation_image = ['top_mosaic_09cm_area34']
    training_image = ['top_mosaic_09cm_area17']
    for image_name in training_image:
        infer_img(input_tensor, logits, keep_probability, sess, image_name, log_dir=log_dir, epoch_num=epoch_num, is_train=True)
    evaluate_dir(log_dir, epoch_num, 'train_acc.txt', 'train')

    for image_name in validation_image:
        infer_img(input_tensor, logits, keep_probability, sess, image_name, log_dir=log_dir, epoch_num=epoch_num)
    evaluate_dir(log_dir, epoch_num, 'val_acc.txt', 'val')


class Batch_manager:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0
    is_new_epoch = False
    seed = 3796

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.__channels = True
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.__channels = False
        self.annotations = np.array(
            [np.expand_dims(self._transform_annotations(filename['annotation']), axis=3) for filename in self.files])
        print (self.images.shape)
        print (self.annotations.shape)

    def _transform(self, filename):
        image = np.load(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])
        return np.array(image).astype(np.float16)

    def _transform_annotations(self, filename):
        return imread(filename, -1)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size, input_tensor, logits, keep_probability, sess, log_dir, is_validation=False):
        self.is_new_epoch = False
        start = self.batch_offset
        self.batch_offset += batch_size
        np.random.seed(self.seed)
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            self.is_new_epoch = True
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")

            if not is_validation:
                infer_and_validate_all_images(input_tensor, logits, keep_probability, sess, str(self.epochs_completed), log_dir)

            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        if start == 0:
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
        end = self.batch_offset
        return self.images[start:end].astype(dtype=np.float32), self.annotations[start:end], self.is_new_epoch

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes].astype(dtype=np.float32), self.annotations[indexes]
