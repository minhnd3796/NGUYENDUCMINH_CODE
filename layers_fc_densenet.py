import numpy as np
import tensorflow as tf

import tensor_utils_fc_densenet as utils


def BN_ReLU_Conv(name, keep_prob, inputs, n_filters, filter_size=3, eps=1e-5):
    depth = np.shape(inputs)[3]
    beta = tf.get_variable(name=name+'_beta', shape=[depth], initializer=tf.constant_initializer(0.0), trainable=False)
    gamma = tf.get_variable(name=name+'_gamma', shape=[depth], initializer=tf.constant_initializer(1.0), trainable=False)
    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
    batch_norm = tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, eps)

    relu_1 = tf.nn.relu(batch_norm)

    W = utils.weight_variable(shape=[filter_size, filter_size, relu_1.get_shape().as_list()[3], n_filters], name=name+"_W")
    bias = utils.bias_variable([n_filters], name=name+"_bias")
    conv = utils.conv2d_basic(relu_1, W, bias=bias)

    relu_2 = tf.nn.relu(conv)
    dropout = tf.nn.dropout(relu_2, keep_prob=keep_prob)
    return dropout


def Transition_Down(inputs, n_filters, keep_prob, name):
    l = BN_ReLU_Conv(name=name, keep_prob=keep_prob, inputs = inputs, n_filters = n_filters, filter_size=1)
    l = utils.max_pool_2x2(l)
    return l


def Transition_Up(skip_connection, block_to_upsample, n_filters_keep, name):
    l = tf.concat(block_to_upsample, axis=3)

    W = utils.weight_variable([3, 3, l.get_shape().as_list()[3], n_filters_keep], name=name+"_W")
    bias = utils.bias_variable([l.get_shape().as_list()[3]], name=name+"_bias")
    de_conv= utils.conv2d_transpose_strided(l, W, bias)

    de_conv = tf.nn.relu(de_conv)

    return tf.concat([de_conv, skip_connection], axis=3 )


if __name__ == '__main__':
    print()
