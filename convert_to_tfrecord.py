import os

import numpy as np
import tensorflow as tf
from scipy.misc import imread

BASE_DIRECTORY = 'ISPRS_semantic_labeling_Vaihingen/'


def get_filename_pair():
    filename_pairs = []
    for filename in os.listdir(BASE_DIRECTORY + 'train'):
        filename_pairs.append((BASE_DIRECTORY + 'train/' + filename,
                               BASE_DIRECTORY + 'train_validate_gt/' + filename.replace('npy', 'png')))
    return filename_pairs


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    tf_records_filename = 'Vaihingen.tfrecords'
    filename_pairs = get_filename_pair()
    print(filename_pairs)
    writer = tf.python_io.TFRecordWriter(tf_records_filename)
    for image_path, annotation_path in filename_pairs:
        image = np.load(image_path)
        annotation = np.array(imread(annotation_path), dtype= np.uint8)
        image_raw = image.tostring()
        annotation_raw = annotation.tostring()
        record = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'annotation_raw': _bytes_feature(annotation_raw)
        }))
        writer.write(record.SerializeToString())
    writer.close()
    print(len(filename_pairs))
    print("Done")
