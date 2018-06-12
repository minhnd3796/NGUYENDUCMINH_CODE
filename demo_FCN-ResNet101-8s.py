import tensorflow as tf
from FCN_8s_ResNet101_5c_v32 import inference as resnet_inference # for 2-skip
from os import environ, mkdir
from sys import argv
from batch_eval_top import create_patch_batch_list, batch_logits_map_inference
import numpy as np
from cv2 import imread, imwrite
from os.path import exists, join
from PIL import Image

IMAGE_SIZE = 224

if __name__ == '__main__':
    environ["CUDA_VISIBLE_DEVICES"] = argv[2]
    # Uncomment if for submission
    """ filename = ['top_mosaic_09cm_area2', 'top_mosaic_09cm_area4', 'top_mosaic_09cm_area6',
                'top_mosaic_09cm_area8', 'top_mosaic_09cm_area10', 'top_mosaic_09cm_area12',
                'top_mosaic_09cm_area14', 'top_mosaic_09cm_area16', 'top_mosaic_09cm_area20',
                'top_mosaic_09cm_area22', 'top_mosaic_09cm_area24', 'top_mosaic_09cm_area27',
                'top_mosaic_09cm_area29', 'top_mosaic_09cm_area31', 'top_mosaic_09cm_area33',
                'top_mosaic_09cm_area35', 'top_mosaic_09cm_area38'] """

    # Comment if for submission
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
    gt_annotation_maps = [None] * num_img_files # Comment if for submission
    pred_annotation_maps = [None] * num_img_files
    top_img = [None] * num_img_files
    num_matches = 0 # Comment if for submission
    num_pixels = 0 # Comment if for submission
    for i in range(num_img_files):
        top_img[i] = imread("../ISPRS_semantic_labeling_Vaihingen/top/" + filename[i] + ".tif")
        logits_maps[i] = np.zeros((top_img[i].shape[0], top_img[i].shape[1], 6), dtype=np.float32)
        gt_annotation_maps[i] = imread("../ISPRS_semantic_labeling_Vaihingen/annotations/" + filename[i] + ".png", -1) # Comment if for submission
        num_pixels += gt_annotation_maps[i].shape[0] * gt_annotation_maps[i].shape[1] # Comment if for submission

    # Accumulate logits maps
    ckpt = tf.train.get_checkpoint_state(argv[1]) # checkpoint directory
    for ckpt_path in ckpt.all_model_checkpoint_paths:
    # for ckpt_path in [tf.train.get_checkpoint_state(argv[1]).model_checkpoint_path]:
        saver.restore(sess, ckpt_path)
        print(">> Restored:", ckpt_path)
        for i in range(num_img_files):
            print(ckpt_path, "inferring", filename[i])
            input_batch_list, coordinate_batch_list, height, width = create_patch_batch_list(filename=filename[i], batch_size=128, num_channels=5)
            current_logits_map = batch_logits_map_inference(input_tensor, logits, keep_probability, sess, is_training, input_batch_list, coordinate_batch_list, height, width)
            logits_maps[i] += current_logits_map

    # Inferring
    for i in range(num_img_files):
        pred_annotation_maps[i] = np.argmax(logits_maps[i], axis=2)
        num_matches += np.sum(pred_annotation_maps[i] == gt_annotation_maps[i]) # Comment if for submission
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
    print("Ensembled Validation Accuracy:", num_matches / num_pixels) # Comment if for submission
