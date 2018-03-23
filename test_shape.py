import numpy as np
import tensorflow as tf
import tensor_utils as utils

def weight_variable(shape, init):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, 14, 14, 512])

resnet101_net = utils.get_model_data('../pretrained_models/imagenet-resnet-101-dag.mat')
weights = np.squeeze(resnet101_net['params'])
kernel = weights[0][1]

W_conv1 = weight_variable([4, 4, 256, 512], kernel)
# b_conv1 = bias_variable([64])
result = conv2d(x, W_conv1)
shape = tf.shape(result)
print(result.shape)
tmp = utils.get_model_data('MatConvNet Tutorials/im_dagnn_tank1.mat')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(type(sess.run(shape, feed_dict={x:tmp['im_'][np.newaxis, :, :, :].astype(np.float32)})))
    # print(shape[1])