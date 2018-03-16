import numpy as np
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, 7, 7, 512])

W_conv1 = weight_variable([7, 7, 512, 4096])
b_conv1 = bias_variable([4096])
result = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(result, feed_dict={x:np.zeros((1, 7, 7, 512))}).shape)