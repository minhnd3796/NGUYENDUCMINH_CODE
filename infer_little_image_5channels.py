from __future__ import print_function

import numpy as np
import tensorflow as tf
from scipy.misc import imread, imsave

import tensor_utils_5_channels as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "../ISPRS_semantic_labeling_Vaihingen", "path to dataset")
tf.flags.DEFINE_string("model_dir", "../ISPRS_semantic_labeling_Vaihingen/imagenet-vgg-verydeep-19.mat", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
MAX_ITERATION = int(1e6 + 1)
NUM_OF_CLASSESS = 6
IMAGE_SIZE = 224

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            if name == 'conv1_1':
                append_channels= np.random.normal(loc=0,scale=0.02,size=(3,3,2,64))
                print(append_channels)
                kernels = np.concatenate((kernels, append_channels), axis=2)
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            else:
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    mean_pixel = np.append(mean_pixel, [30.6986130799, 283.307])
    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def infer_little_img(input_image_path,patch_size=224,stride_ver=112,stride_hor=112):
    input_image= imread(input_image_path)
    dsm_image= imread(input_image_path.replace('top','dsm').replace('_mosaic','').replace('area','matching_area'))
    ndsm_image= imread(input_image_path.replace('top/','ndsm/').replace('top','dsm').replace('_mosaic','')
                       .replace('area','matching_area').replace('.tif','_normalized.jpg'))
    dsm_image= np.expand_dims(dsm_image,axis=2)
    ndsm_image= np.expand_dims(ndsm_image,axis=2)
    height = np.shape(input_image)[0]
    width = np.shape(input_image)[1]
    output_image = np.zeros_like(input_image)
    input_image= np.concatenate((input_image,ndsm_image,dsm_image),axis=2)
    output_map = np.zeros((height, width, 6), dtype=np.float32)
    number_of_vertical_points = (height - patch_size) // stride_ver + 1
    number_of_horizontial_points = (width - patch_size) // stride_hor + 1
    sess= tf.Session()
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 5], name="input_image")
    _, logits = inference(image, keep_probability)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    input_image= np.expand_dims(input_image,axis=0)
    for i in range(number_of_vertical_points):
        for j in range(number_of_horizontial_points):
            current_patch = input_image[:,i * stride_ver:i * stride_ver + patch_size,
                            j * stride_hor:j * stride_hor + patch_size, :]
            logits_result = sess.run(logits, feed_dict={image: current_patch, keep_probability: 1.0})
            logits_result = tf.squeeze(logits_result)
            patch_result= sess.run(logits_result)
            output_map[i * stride_ver:i * stride_ver + patch_size, j * stride_hor:j * stride_hor + patch_size,
            :] += patch_result
            print('stage 1: i='+str(i)+"; j="+str(j))
    for i in range(number_of_vertical_points):
        current_patch= input_image[:,i*stride_ver:i*stride_ver+patch_size,width-patch_size:width,:]
        logits_result = sess.run(logits, feed_dict={image: current_patch, keep_probability: 1.0})
        logits_result = tf.squeeze(logits_result)
        patch_result = sess.run(logits_result)
        output_map[i*stride_ver:i*stride_ver+patch_size,width-patch_size:width,:]+=patch_result
        print('stage 2: i=' + str(i) + "; j=" + str(j))
    for i in range(number_of_horizontial_points):
        current_patch= input_image[:,height-patch_size:height,i*stride_hor:i*stride_hor+patch_size,:]
        logits_result = sess.run(logits, feed_dict={image: current_patch, keep_probability: 1.0})
        logits_result = tf.squeeze(logits_result)
        patch_result = sess.run(logits_result)
        output_map[height-patch_size:height,i*stride_hor:i*stride_hor+patch_size,:]+=patch_result
        print('stage 3: i=' + str(i) + "; j=" + str(j))
    current_patch = input_image[:,height - patch_size:height, width - patch_size:width, :]
    logits_result = sess.run(logits, feed_dict={image: current_patch, keep_probability: 1.0})
    logits_result = tf.squeeze(logits_result)
    patch_result = sess.run(logits_result)
    output_map[height - patch_size:height, width - patch_size:width, :] += patch_result
    predict_annotation_image = np.argmax(output_map, axis=2)
    print(np.shape(predict_annotation_image))
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
    return output_image

if __name__ == "__main__":
    #tf.app.run()
    imsave("top_mosaic_09cm_area38.tif",
           infer_little_img("../ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area38.tif"))


# 2
# 4
# 6
# 8
# 10
# 12
# 14
# 16
# 20
# 22
# 24
# 27
# 29
# 31
# 33
# 35
# 38