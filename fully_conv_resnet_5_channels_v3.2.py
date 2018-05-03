from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from six.moves import xrange
import datetime
import Batch_manager_5channels as dataset
import data_reader_5channels as reader
import tensor_utils_5_channels as utils
from infer_imagenet_resnet_101_5chan_v31 import resnet101_net
from sys import argv
from os.path import join

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "32", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "../logs-resnet101_5channels_v3.2/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "../ISPRS_semantic_labeling_Vaihingen", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "../pretrained_models/imagenet-resnet-101-dag.mat", "Path to model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/imagenet-resnet-101-dag.mat'

# MAX_ITERATION = int(1e6 + 1)
MAX_ITERATION = int(153600 + 1) # 25 epochs
NUM_OF_CLASSES = 6
IMAGE_SIZE = 224

def inference(image, keep_prob, is_training):
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
    mean_pixel[:, :, 3] = np.ones((IMAGE_SIZE, IMAGE_SIZE)) * 30.69861307993539
    mean_pixel[:, :, 4] = np.ones((IMAGE_SIZE, IMAGE_SIZE)) * 284.9702

    normalised_img = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        net = resnet101_net(normalised_img, weights, keep_prob, is_training)
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

def _decay(weight_decay):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find(r'W') > 0 or var.op.name.find(r'w') > 0 or var.op.name.find(r'filter') > 0:
            costs.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
    return tf.multiply(weight_decay, tf.add_n(costs))

def build_session(cuda_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 5], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    is_training = tf.placeholder(tf.bool, name="is_training")
    pred_annotation, logits = inference(image, keep_probability, is_training)
    annotation_64 = tf.cast(annotation, dtype=tf.int64)
    # calculate accuracy for batch.
    cal_acc = tf.equal(pred_annotation, annotation_64)
    cal_acc = tf.cast(cal_acc, dtype=tf.int8)
    acc = tf.count_nonzero(cal_acc) / (FLAGS.batch_size * IMAGE_SIZE * IMAGE_SIZE)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

    """ weight_decay = 1e-4
    l2_loss = _decay(weight_decay)
    cross_entropy = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[3]), name="entropy")))
    loss = l2_loss + cross_entropy
    loss_summary = tf.summary.scalar("entropy_with_l2", loss) """

    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[3]), name="entropy")))
    loss_summary = tf.summary.scalar("entropy", loss)

    # summary accuracy in tensorboard
    acc_summary = tf.summary.scalar("accuracy", acc)
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)
    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

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
    return image, logits, is_training, keep_probability, sess, annotation, train_op, loss, acc, loss_summary, acc_summary, saver, pred_annotation, train_writer, validation_writer

def main(argv=None):
    np.random.seed(3796)
    image, logits, is_training, keep_probability, sess, annotation, train_op, loss, acc, loss_summary, acc_summary, saver, pred_annotation, train_writer, validation_writer = build_session(argv[1])

    print("Setting up image reader...")
    train_records, valid_records = reader.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': False, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.Batch_manager(train_records, image_options)
    validation_dataset_reader = dataset.Batch_manager(valid_records, image_options)

    """ os.environ["CUDA_VISIBLE_DEVICES"] = argv[1]
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print("Setting up Saver...")
    saver = tf.train.Saver()
    # train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    # validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...") """

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(saver, FLAGS.batch_size, image, logits, keep_probability, sess, is_training, FLAGS.logs_dir)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.5, is_training: True}
            tf.set_random_seed(3796 + itr) # get deterministicly random dropouts
            sess.run(train_op, feed_dict=feed_dict)

            if itr % 50 == 0:
                feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 1.0, is_training: False}
                train_loss, train_acc, summary_loss, summary_acc = sess.run([loss, acc, loss_summary, acc_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss: %g, Train_acc: %g" % (itr, train_loss, train_acc))
                with open(join(FLAGS.logs_dir, 'iter_train_loss.csv'), 'a') as f:
                    f.write(str(itr) + ',' + str(train_loss) + '\n')
                with open(join(FLAGS.logs_dir, 'iter_train_acc.csv'), 'a') as f:
                    f.write(str(itr) + ',' + str(train_acc) + '\n')
                train_writer.add_summary(summary_loss, itr)
                train_writer.add_summary(summary_acc, itr)
            if itr % 600 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(saver, FLAGS.batch_size, image, logits, keep_probability, sess, is_training, FLAGS.logs_dir, is_validation=True)
                valid_loss, valid_acc, summary_loss, summary_acc = sess.run([loss, acc, loss_summary, acc_summary],
                                                feed_dict={image: valid_images, annotation: valid_annotations,
                                                            keep_probability: 1.0, is_training: False})
                validation_writer.add_summary(summary_loss, itr)
                validation_writer.add_summary(summary_acc, itr)
                print("%s ---> Validation_loss: %g , Validation Accuracy: %g" % (datetime.datetime.now(), valid_loss, valid_acc))
                with open(join(FLAGS.logs_dir, 'iter_val_loss.csv'), 'a') as f:
                    f.write(str(itr) + ',' + str(valid_loss) + '\n')
                with open(join(FLAGS.logs_dir, 'iter_val_acc.csv'), 'a') as f:
                    f.write(str(itr) + ',' + str(valid_acc) + '\n')
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0, is_training: False})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            print(valid_images[itr].astype(np.uint8).shape)
            utils.save_image(valid_images[itr, :, :, :3].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(itr))
            print(valid_annotations[itr].astype(np.uint8).shape)
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(itr))
            print(pred[itr].astype(np.uint8).shape)
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(itr))
            print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run()
