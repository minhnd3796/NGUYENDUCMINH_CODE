import tensorflow as tf
from fully_conv_resnet_potsdam import inference as resnet_inference # for 3-skip
from os import environ, mkdir
from sys import argv
from batch_eval_potsdam import create_patch_batch_list, batch_logits_map_inference
import numpy as np
from cv2 import imread, imwrite
from os.path import exists, join
from PIL import Image

IMAGE_SIZE = 224

if __name__ == '__main__':
    environ["CUDA_VISIBLE_DEVICES"] = argv[2]
    # Uncomment if for submission
    filename = ['top_potsdam_2_13_label', 'top_potsdam_2_14_label', 'top_potsdam_3_13_label',
                'top_potsdam_3_14_label', 'top_potsdam_4_13_label', 'top_potsdam_4_14_label',
                'top_potsdam_4_15_label', 'top_potsdam_5_13_label', 'top_potsdam_5_14_label',
                'top_potsdam_5_15_label', 'top_potsdam_6_13_label', 'top_potsdam_6_14_label',
                'top_potsdam_6_15_label', 'top_potsdam_7_13_label']

    # Comment if for submission
    """ filename = ["top_potsdam_2_11_label", "top_potsdam_3_12_label", "top_potsdam_4_10_label",
                "top_potsdam_6_11_label", "top_potsdam_7_12_label"] """
    

    is_training = tf.placeholder(tf.bool, name="is_training")
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    input_tensor = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 6], name="input_image")
    _, logits = resnet_inference(input_tensor, keep_probability, is_training)

    sess = tf.Session()
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Init logits maps
    num_img_files = len(filename)
    logits_maps = [None] * num_img_files
    # gt_annotation_maps = [None] * num_img_files # Comment if for submission
    pred_annotation_maps = [None] * num_img_files
    top_img = [None] * num_img_files
    # num_matches = 0 # Comment if for submission
    # num_pixels = 0 # Comment if for submission
    for i in range(num_img_files):
        top_img[i] = np.load("../ISPRS_semantic_labeling_Potsdam/npy_6_channel/" + filename[i].replace('label', 'RGBIR') + ".npy")
        logits_maps[i] = np.zeros((top_img[i].shape[0], top_img[i].shape[1], 6), dtype=np.float32)
        # gt_annotation_maps[i] = imread("../ISPRS_semantic_labeling_Potsdam/annotations/" + filename[i] + ".png", -1) # Comment if for submission
        # num_pixels += gt_annotation_maps[i].shape[0] * gt_annotation_maps[i].shape[1] # Comment if for submission

    # Accumulate logits maps
    ckpt = tf.train.get_checkpoint_state(argv[1]) # checkpoint directory
    for ckpt_path in ckpt.all_model_checkpoint_paths:
    # for ckpt_path in [tf.train.get_checkpoint_state(argv[1]).model_checkpoint_path]:
        saver.restore(sess, ckpt_path)
        print(">> Restored:", ckpt_path)
        for i in range(num_img_files):
            print(ckpt_path, "inferring", filename[i])
            input_batch_list, coordinate_batch_list, height, width = create_patch_batch_list(filename=filename[i], batch_size=512, num_channels=6)
            current_logits_map = batch_logits_map_inference(input_tensor, logits, keep_probability, sess, is_training, input_batch_list, coordinate_batch_list, height, width)
            logits_maps[i] += current_logits_map

    # Inferring
    for i in range(num_img_files):
        pred_annotation_maps[i] = np.argmax(logits_maps[i], axis=2)
        # num_matches += np.sum(pred_annotation_maps[i] == gt_annotation_maps[i]) # Comment if for submission
        height = pred_annotation_maps[i].shape[0]
        width = pred_annotation_maps[i].shape[1]
        output_image = np.zeros((height, width, 3), dtype=np.uint8)

        print("Generating", filename[i] + '.tif......')
        for y in range(height):
            for x in range(width):
                if pred_annotation_maps[i][y, x]==0:
                    output_image[y, x]=np.array([255,255,255])
                elif pred_annotation_maps[i][y, x]==1:
                    output_image[y, x]=np.array([0,0,255])
                elif pred_annotation_maps[i][y, x]==2:
                    output_image[y, x]=np.array([0,255,255])
                elif pred_annotation_maps[i][y, x]==3:
                    output_image[y, x]=np.array([0,255,0])
                elif pred_annotation_maps[i][y, x]==4:
                    output_image[y, x]=np.array([255,255,0])
                elif pred_annotation_maps[i][y, x]==5:
                    output_image[y, x]=np.array([255,0,0])
        if not exists(join(argv[1], 'submission_cv2')):
            mkdir(join(argv[1], 'submission_cv2'))
        if not exists(join(argv[1], 'submission_PIL')):
            mkdir(join(argv[1], 'submission_PIL'))
        img = Image.fromarray(output_image)
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
        img.save(join(argv[1], 'submission_PIL', filename[i] + '_class.tif'))
        imwrite(join(argv[1], 'submission_cv2', filename[i] + '_class.png'), output_image)

    # Print ensemble accuracy
    # print("Ensembled Validation Accuracy:", num_matches / num_pixels) # Comment if for submission
