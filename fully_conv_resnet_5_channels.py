from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from six.moves import xrange

import datetime

import Batch_manager_5channels as dataset
import data_reader_5channels as reader
import tensor_utils_5_channels as utils

from infer_imagenet_resnet_101_5chan import resnet101_net

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "32", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "../logs-resnet101/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "../ISPRS_semantic_labeling_Vaihingen", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "../pretrained_models/imagenet-resnet-101-dag.mat",
                       "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/imagenet-resnet-101-dag.mat'

MAX_ITERATION = int(1e6 + 1)
NUM_OF_CLASSES = 6
IMAGE_SIZE = 224
VALIDATE_IMAGES = ["top_mosaic_09cm_area7.png","top_mosaic_09cm_area17.png","top_mosaic_09cm_area23.png","top_mosaic_09cm_area37.png"]


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
    print(">> Setting up resnet-101 pretrained layers ...")

    resnet101_model = utils.get_model_data(FLAGS.model_dir)
    weights = np.squeeze(resnet101_model['params'])
    mean_pixel_init = resnet101_model['meta'][0][0][2][0][0][2]
    mean_pixel = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 5))
    mean_pixel[:, :, 0] = mean_pixel_init[:, :, 0]
    mean_pixel[:, :, 1] = mean_pixel_init[:, :, 1]
    mean_pixel[:, :, 2] = mean_pixel_init[:, :, 2]
    mean_pixel[:, :, 3] = np.ones((IMAGE_SIZE, IMAGE_SIZE)) * 30.6986130799
    mean_pixel[:, :, 4] = np.ones((IMAGE_SIZE, IMAGE_SIZE)) * 284.97018

    normalised_img = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        net = resnet101_net(normalised_img, weights)
        last_layer = net["res5c_relu"]

        fc_filter = utils.weight_variable([1, 1, 2048, NUM_OF_CLASSES], name="fc_filter")
        fc_bias = utils.bias_variable([NUM_OF_CLASSES], name="fc_bias")
        fc = tf.nn.bias_add(tf.nn.conv2d(last_layer, fc_filter, strides=[1, 1, 1, 1], padding="SAME"), fc_bias, name='fc')

        # now to upscale to actual image size
        deconv_shape1 = net["res4b22_relu"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSES], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(fc, W_t1, b_t1, output_shape=tf.shape(net["res4b22_relu"]))
        fuse_1 = tf.add(conv_t1, net["res4b22_relu"], name="fuse_1")

        deconv_shape2 = net["res3b3_relu"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(net["res3b3_relu"]))
        fuse_2 = tf.add(conv_t2, net["res3b3_relu"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSES])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSES, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSES], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, axis=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 5], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    pred_annotation, logits = inference(image, keep_probability)
    annotation_64 = tf.cast(annotation, dtype=tf.int64)
    # calculate accuracy for batch.
    cal_acc = tf.equal(pred_annotation, annotation_64)
    cal_acc = tf.cast(cal_acc, dtype=tf.int8)
    acc = tf.count_nonzero(cal_acc) / (FLAGS.batch_size * IMAGE_SIZE * IMAGE_SIZE)

    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation,
                                                                                            squeeze_dims=[3]),
                                                                          name="entropy")))
    loss_summary=tf.summary.scalar("entropy", loss)

    # summary accuracy in tensorboard
    acc_summary=tf.summary.scalar("accuracy", acc)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records = reader.read_dataset_resnet101(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': False, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.Batch_manager(train_records, image_options)
    validation_dataset_reader = dataset.Batch_manager(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    for itr in xrange(MAX_ITERATION):
        train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
        feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.75}
        sess.run(train_op, feed_dict=feed_dict)

        if itr % 50 == 0:
            train_loss, train_acc, summary_loss, summary_acc = sess.run([loss, acc, loss_summary, acc_summary], feed_dict=feed_dict)
            print("Step: %d, Train_loss: %g, Train_acc: %g" % (itr, train_loss, train_acc))
            train_writer.add_summary(summary_loss, itr)
            train_writer.add_summary(summary_acc, itr)
        if itr % 500 == 0:
            valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
            valid_loss, valid_acc, summary_loss, summary_acc = sess.run([loss, acc, loss_summary, acc_summary],
                                             feed_dict={image: valid_images, annotation: valid_annotations,
                                                        keep_probability: 1.0})
            validation_writer.add_summary(summary_loss, itr)
            validation_writer.add_summary(summary_acc, itr)
            print("%s ---> Validation_loss: %g , Validation Accuracy: %g" % (
                datetime.datetime.now(), valid_loss, valid_acc))
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)


if __name__ == "__main__":
    tf.app.run()
