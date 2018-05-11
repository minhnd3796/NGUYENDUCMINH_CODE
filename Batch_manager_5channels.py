import numpy as np
from cv2 import imread, imwrite
import tensorflow as tf
from os.path import exists, join
from os import mkdir
from batch_eval_top import eval_dir
from batch_eval_potsdam import eval_dir_potsdam

class Batch_manager:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0
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
        # self.images = np.array([np.load(filename) for filename in self.files])        
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

    def next_batch(self, saver, batch_size, input_tensor, logits, keep_probability, sess, is_training, log_dir, encoding_keep_prob=None, is_validation=False):
        start = self.batch_offset
        self.batch_offset += batch_size
        np.random.seed(self.seed)
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            saver.save(sess, log_dir + "model.ckpt", self.epochs_completed)
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            """ if not is_validation:
                eval_dir(input_tensor, logits, keep_probability, sess, is_training, batch_size, log_dir, self.epochs_completed, encoding_keep_prob=encoding_keep_prob, is_validation=False, num_channels=5)
                eval_dir(input_tensor, logits, keep_probability, sess, is_training, batch_size, log_dir, self.epochs_completed, encoding_keep_prob=encoding_keep_prob, is_validation=True, num_channels=5) """
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        if start == 0:
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
        end = self.batch_offset
        return self.images[start:end].astype(dtype=np.float32), self.annotations[start:end]

    def next_batch_15(self, saver, batch_size, input_tensor, logits, keep_probability, sess, is_training, log_dir, encoding_keep_prob=None, is_validation=False):
        start = self.batch_offset
        self.batch_offset += batch_size
        np.random.seed(self.seed)
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            saver.save(sess, log_dir + "model.ckpt", self.epochs_completed)
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            if not is_validation:
                eval_dir(input_tensor, logits, keep_probability, sess, is_training, batch_size, log_dir, self.epochs_completed, encoding_keep_prob=encoding_keep_prob, is_validation=False, num_channels=15)
                eval_dir(input_tensor, logits, keep_probability, sess, is_training, batch_size, log_dir, self.epochs_completed, encoding_keep_prob=encoding_keep_prob, is_validation=True, num_channels=15)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        if start == 0:
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
        end = self.batch_offset
        return self.images[start:end].astype(dtype=np.float32), self.annotations[start:end]
    
    def next_batch_potsdam(self, saver, batch_size, input_tensor, logits, keep_probability, sess, is_training, log_dir, encoding_keep_prob=None, is_validation=False):
        start = self.batch_offset
        self.batch_offset += batch_size
        np.random.seed(self.seed)
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            saver.save(sess, log_dir + "model.ckpt", self.epochs_completed)
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            if not is_validation:
                eval_dir_potsdam(input_tensor, logits, keep_probability, sess, is_training, batch_size, log_dir, self.epochs_completed, encoding_keep_prob=encoding_keep_prob, is_validation=False)
                eval_dir_potsdam(input_tensor, logits, keep_probability, sess, is_training, batch_size, log_dir, self.epochs_completed, encoding_keep_prob=encoding_keep_prob, is_validation=True)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        if start == 0:
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
        end = self.batch_offset
        return self.images[start:end].astype(dtype=np.float32), self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes].astype(dtype=np.float32), self.annotations[indexes]
