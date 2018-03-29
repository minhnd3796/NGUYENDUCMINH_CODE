from __future__ import print_function

import os

import datetime

import numpy as np
import tensorflow as tf
from six.moves import xrange

import Batch_manager as dataset
import data_reader as scene_parsing
import tensor_utils as utils

from infer_imagenet_resnet_101 import resnet101_net

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "256", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "../logs-resnet101/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "../ISPRS_semantic_labeling_Vaihingen", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-5", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "../pretrained_models/imagenet-resnet-101-dag.mat",
                       "Path to resnet-101 model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/imagenet-resnet-101-dag.mat'

MAX_ITERATION = int(1e6 + 1)
NUM_OF_CLASSES = 6
IMAGE_SIZE = 224

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
    mean_pixel = resnet101_model['meta'][0][0][2][0][0][2]

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
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    pred_annotation, logits = inference(image, keep_probability)
    annotation_64 = tf.cast(annotation, dtype=tf.int64)
    # calculate accuracy for batch.
    cal_acc = tf.equal(pred_annotation, annotation_64)
    cal_acc = tf.cast(cal_acc, dtype=tf.int8)
    acc = tf.count_nonzero(cal_acc)/(FLAGS.batch_size * IMAGE_SIZE *IMAGE_SIZE)

    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation,
                                                                                            squeeze_dims=[3]),
                                                                          name="entropy")))
    tf.summary.scalar("entropy", loss)

    # summary accuracy in tensorboard
    tf.summary.scalar("accuracy", acc)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print(">> Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print(">> Setting up image reader...")
    print(FLAGS.data_dir)
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print("No. of train records:", len(train_records))
    print("No. of test records", len(valid_records))

    print(">> Setting up dataset reader")
    image_options = {'resize': False, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.Batch_manager(train_records, image_options)
    validation_dataset_reader = dataset.Batch_manager(valid_records, image_options)

    sess = tf.Session()

    print(">> Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(">> Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.75}
            sess.run(train_op, feed_dict=feed_dict)

            if itr % 50 == 0:
                train_loss, train_acc, summary_str = sess.run([loss, acc, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss: %g, Train_acc: %g" % (itr, train_loss, train_acc))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss, valid_acc = sess.run([loss, acc], feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g , Validation Accuracy: %g" % (datetime.datetime.now(), valid_loss, valid_acc))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5 + itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5 + itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5 + itr))
            print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run()
