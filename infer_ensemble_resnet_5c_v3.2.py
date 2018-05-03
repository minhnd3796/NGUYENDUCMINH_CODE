import tensorflow as tf
from fully_conv_resnet_5_channels_v3 import inference as resnet_inference
from os import environ
from sys import argv
from batch_eval_top import create_patch_batch_list, batch_logits_map_inference
import numpy as np
from cv2 import imread

IMAGE_SIZE = 224

if __name__ == '__main__':
    environ["CUDA_VISIBLE_DEVICES"] = argv[2]
    """ filename = ['top_mosaic_09cm_area2', 'top_mosaic_09cm_area4', 'top_mosaic_09cm_area6',
                'top_mosaic_09cm_area8', 'top_mosaic_09cm_area10', 'top_mosaic_09cm_area12',
                'top_mosaic_09cm_area14', 'top_mosaic_09cm_area16', 'top_mosaic_09cm_area20',
                'top_mosaic_09cm_area22', 'top_mosaic_09cm_area24', 'top_mosaic_09cm_area27',
                'top_mosaic_09cm_area29', 'top_mosaic_09cm_area31', 'top_mosaic_09cm_area33',
                'top_mosaic_09cm_area35', 'top_mosaic_09cm_area38'] """

    filename = ['top_mosaic_09cm_area7', 'top_mosaic_09cm_area17', 'top_mosaic_09cm_area23', 'top_mosaic_09cm_area37']

    is_training = tf.placeholder(tf.bool, name="is_training")
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    input_tensor = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 5], name="input_image")
    _, logits = resnet_inference(input_tensor, keep_probability, is_training)

    sess = tf.Session()
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Init logits maps
    num_img_files = len(filename)
    logits_maps = [None] * num_img_files
    gt_annotation_maps = [None] * num_img_files
    pred_annotation_maps = [None] * num_img_files
    num_matches = 0
    num_pixels = 0
    for i in range(num_img_files):
        gt_annotation_maps[i] = imread("../ISPRS_semantic_labeling_Vaihingen/annotations/" + filename[i] + ".png", -1)
        logits_maps[i] = np.zeros((gt_annotation_maps[i].shape[0], gt_annotation_maps[i].shape[1], 6), dtype=np.float32)
        num_pixels += gt_annotation_maps[i].shape[0] * gt_annotation_maps[i].shape[1]

    # Accumulate logits maps
    ckpt = tf.train.get_checkpoint_state(argv[1]) # checkpoint directory
    for ckpt_path in ckpt.all_model_checkpoint_paths:
    # for ckpt_path in ckpt.all_model_checkpoint_paths:
        saver.restore(sess, ckpt_path)
        for i in range(num_img_files):
            input_batch_list, coordinate_batch_list, height, width = create_patch_batch_list(filename=filename[i], batch_size=128, num_channels=5)
            current_logits_map = batch_logits_map_inference(input_tensor, logits, keep_probability, sess, is_training, input_batch_list, coordinate_batch_list, height, width)
            logits_maps[i] += current_logits_map
    
    # Inferring
    for i in range(num_img_files):
        pred_annotation_maps[i] = np.argmax(logits_maps[i], axis=2)
        num_matches += np.sum(pred_annotation_maps[i] == gt_annotation_maps[i])
    
    # Print ensemble accuracy
    print("Ensembled Validation Accuracy:", num_matches / num_pixels)
    