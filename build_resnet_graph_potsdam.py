from os import environ
from sys import argv
import tensor_utils_5_channels as utils
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from scipy.misc import imread
from skimage.transform import resize

def _input():
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    is_training = tf.placeholder(tf.bool, name="is_training")
    return input_tensor, keep_prob, is_training

def create_branch_layer_names(res_num, res_type, branch_type, branch_order=''):
    if branch_type == 1 or branch_order == 'c':
        name_list = [None] * 2
        name_list[0] = 'res' + str(res_num) + res_type + '_branch' + str(branch_type) + branch_order
        name_list[1] = 'bn' + str(res_num) + res_type + '_branch' + str(branch_type) + branch_order
    else:
        name_list = [None] * 3
        name_list[0] = 'res' + str(res_num) + res_type + '_branch' + str(branch_type) + branch_order
        name_list[1] = 'bn' + str(res_num) + res_type + '_branch' + str(branch_type) + branch_order
        name_list[2] = 'res' + str(res_num) + res_type + '_branch' + str(branch_type) + branch_order + '_relu'
    return name_list

def create_param_names_from_layers(layer_names):
    param_list = [None] * 4
    param_list[0] = layer_names[0] + '_filter'
    param_list[1] = layer_names[1] + '_mult'
    param_list[2] = layer_names[1] + '_bias'
    param_list[3] = layer_names[1] + '_moments'
    return param_list

def construct_batch_normalisation_block(current, net, weights, start_weight_index, param_names, layer_names, is_training):
    scale = weights[start_weight_index + 1][1].reshape(-1)
    scale = utils.get_variable(scale, name=param_names[1])

    offset = weights[start_weight_index + 2][1].reshape(-1)
    offset = utils.get_variable(offset, name=param_names[2])

    mean = weights[start_weight_index + 3][1][:, 0].reshape(-1)
    mean = utils.get_variable(mean, name=param_names[3] + '_mean')

    variance = weights[start_weight_index + 3][1][:, 1].reshape(-1)
    variance = utils.get_variable(variance * variance, name=param_names[3] + '_variance')

    batch_mean, batch_var = tf.nn.moments(current, [0, 1, 2], name='batch_moments')
    decay = 1 - weights[start_weight_index + 3][2][0][0]
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # decay = 0.9999

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

    current = tf.nn.batch_normalization(current, mean, variance, offset, scale, 1e-5, name=layer_names[1])

    net[layer_names[1]] = current

    return current, net

def construct_conv_bn_block(current, net, weights, start_weight_index, param_names, layer_names, stride, keep_prob, is_training):
    # conv
    kernel = weights[start_weight_index][1]
    if param_names[0] == 'conv1_filter':
        np.random.seed(3796)
        appended_kernel = np.random.normal(loc=0, scale=0.02, size=(7, 7, 3, 64)) # for 6 channels of Potsdam
        kernel = np.concatenate((kernel, appended_kernel), axis=2)
    else:
        kernel = utils.get_variable(kernel, name=param_names[0])
    if stride == 1:
        current = tf.nn.conv2d(current, kernel, strides=[1, 1, 1, 1], padding='SAME', name=layer_names[0])
    else:
        kernel_size = kernel.shape[0]
        padding = int((int(kernel_size) - 1) / 2)
        current = tf.nn.conv2d(
            tf.pad(current, tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]])),
            kernel, strides=[1, stride, stride, 1], padding='VALID', name=layer_names[0])
    # net[layer_names[0]] = tf.nn.dropout(current, keep_prob=keep_prob)

    # bn
    # print(layer_names)
    # print(param_names)
    # print()
    current, net = construct_batch_normalisation_block(current, net, weights, start_weight_index, param_names, layer_names, is_training)
    return current, net

def construct_conv_bn_relu_block(current, net, weights, start_weight_index, param_names, layer_names, stride, keep_prob, is_training):
    # conv_bn
    current, net = construct_conv_bn_block(current, net, weights, start_weight_index, param_names, layer_names, stride, keep_prob, is_training)
    # relu
    current = tf.nn.relu(current, name=layer_names[2])
    net[layer_names[2]] = current
    return current, net

def construct_branch1_block(res_num, input_tensor, net, weights, start_param_index, first_conv_stride, keep_prob, is_training):
    branch1_layer_names = create_branch_layer_names(res_num, res_type='a', branch_type=1)
    branch1_param_names = create_param_names_from_layers(branch1_layer_names)
    current, net = construct_conv_bn_block(input_tensor, net, weights, start_param_index, branch1_param_names, branch1_layer_names, first_conv_stride, keep_prob, is_training)
    return current, net

def construct_branch2_block(res_num, res_type, input_tensor, net, weights, start_param_index, first_conv_stride, keep_prob, is_training):
    branch2a_layer_names = create_branch_layer_names(res_num, res_type, branch_type=2, branch_order='a')
    branch2a_param_names = create_param_names_from_layers(branch2a_layer_names)
    current, net = construct_conv_bn_relu_block(input_tensor, net, weights, start_param_index, branch2a_param_names, branch2a_layer_names, first_conv_stride, keep_prob, is_training)
    start_param_index += 4

    branch2b_layer_names = create_branch_layer_names(res_num, res_type, branch_type=2, branch_order='b')
    branch2b_param_names = create_param_names_from_layers(branch2b_layer_names)
    current, net = construct_conv_bn_relu_block(current, net, weights, start_param_index, branch2b_param_names, branch2b_layer_names, 1, keep_prob, is_training)
    start_param_index += 4

    branch2c_layer_names = create_branch_layer_names(res_num, res_type, branch_type=2, branch_order='c')
    branch2c_param_names = create_param_names_from_layers(branch2c_layer_names)
    current, net = construct_conv_bn_block(current, net, weights, start_param_index, branch2c_param_names, branch2c_layer_names, 1, keep_prob, is_training)
    start_param_index += 4

    return current, net

def construct_res_xa_block(res_num, input_tensor, net, weights, start_param_index, keep_prob, is_training, down_sample=True):
    # resxa_branch1 block
    if down_sample == True:
        first_conv_stride = 2
    else:
        first_conv_stride = 1
    current, net = construct_branch1_block(res_num, input_tensor, net, weights, start_param_index, first_conv_stride, keep_prob, is_training)
    bn_branch1 = current
    start_param_index += 4

    # resxa_branch2 block
    current, net = construct_branch2_block(res_num, 'a', input_tensor, net, weights, start_param_index, first_conv_stride, keep_prob, is_training)
    start_param_index += 12

    current = tf.add(bn_branch1, current, name='res' + str(res_num) + 'a')
    net['res' + str(res_num) + 'a'] = current
    current = tf.nn.relu(current, name='res' + str(res_num) + 'a_relu')
    net['res' + str(res_num) + 'a_relu'] = current

    return current, net, start_param_index

def construct_res_xxx_block(res_num, res_type, input_tensor, net, weights, start_param_index, keep_prob, is_training):
    current, net = construct_branch2_block(res_num, res_type, input_tensor, net, weights, start_param_index, 1, keep_prob, is_training)
    start_param_index += 12

    current = tf.add(input_tensor, current, name='res' + str(res_num) + res_type)
    net['res' + str(res_num) + res_type] = current
    current = tf.nn.relu(current, name='res' + str(res_num) + res_type + '_relu')
    if 'res' + str(res_num) + res_type + '_relu' == 'res5c_relu':
        net['res' + str(res_num) + res_type + '_relu'] = tf.nn.dropout(current, keep_prob=keep_prob)
    else:
        net['res' + str(res_num) + res_type + '_relu'] = current

    return current, net, start_param_index

def resnet101_net(image, weights, keep_prob, is_training):
    net = {}
    current = image
    start_param_index = 0

    # conv1 block
    conv1_layer_names = ['conv1', 'bn_conv1', 'conv1_relu', 'pool1']
    conv1_param_names = create_param_names_from_layers(conv1_layer_names)
    current, net = construct_conv_bn_relu_block(current, net, weights, start_param_index, conv1_param_names, conv1_layer_names, 2, keep_prob, is_training)
    current = tf.nn.max_pool(current, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=conv1_layer_names[3])
    net[conv1_layer_names[3]] = current
    start_param_index += 4

    current, net, start_param_index = construct_res_xa_block(2, current, net, weights, start_param_index, keep_prob, is_training, down_sample=False)
    current, net, start_param_index = construct_res_xxx_block(2, 'b', current, net, weights, start_param_index, keep_prob, is_training)
    current, net, start_param_index = construct_res_xxx_block(2, 'c', current, net, weights, start_param_index, keep_prob, is_training)

    current, net, start_param_index = construct_res_xa_block(3, current, net, weights, start_param_index, keep_prob, is_training, down_sample=True)
    for i in range(1, 4):
        current, net, start_param_index = construct_res_xxx_block(3, 'b' + str(i), current, net, weights, start_param_index, keep_prob, is_training)

    current, net, start_param_index = construct_res_xa_block(4, current, net, weights, start_param_index, keep_prob, is_training, down_sample=True)
    for i in range(1, 23):
        current, net, start_param_index = construct_res_xxx_block(4, 'b' + str(i), current, net, weights, start_param_index, keep_prob, is_training)

    current, net, start_param_index = construct_res_xa_block(5, current, net, weights, start_param_index, keep_prob, is_training, down_sample=True)
    current, net, start_param_index = construct_res_xxx_block(5, 'b', current, net, weights, start_param_index, keep_prob, is_training)
    current, net, start_param_index = construct_res_xxx_block(5, 'c', current, net, weights, start_param_index, keep_prob, is_training)

    current = tf.nn.avg_pool(current, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool5')
    net['pool5'] = current

    fc1000_kernel = utils.get_variable(weights[start_param_index][1], name='fc1000_filter')
    fc1000_bias = utils.get_variable(weights[start_param_index + 1][1].reshape(-1), name='fc1000_bias')
    current = tf.nn.bias_add(tf.nn.conv2d(current, fc1000_kernel, strides=[1, 1, 1, 1], padding="VALID"), fc1000_bias, name='fc1000')
    net['fc1000'] = current

    current = tf.nn.softmax(current, name='prob')
    net['prob'] = current

    return net

def inference(x, weights, keep_prob, is_training):
    with tf.variable_scope("inference"):
        image_net = resnet101_net(x, weights, keep_prob, is_training)
        prediction = tf.argmax(image_net['prob'][0][0][0])

    return prediction, image_net

def main(argv=None):
    environ["CUDA_VISIBLE_DEVICES"] = argv[1]
    resnet101_net = utils.get_model_data('../pretrained_models/imagenet-resnet-101-dag.mat')
    weights = np.squeeze(resnet101_net['params'])

    img = imread(argv[2])
    mean = resnet101_net['meta'][0][0][2][0][0][2]
    resized_img = resize(img, (224, 224), preserve_range=True, mode='reflect')
    normalised_img = utils.process_image(resized_img, mean)

    input_tensor, keep_prob, is_training = _input()
    predicted_class, image_net = inference(input_tensor, weights, keep_prob, is_training)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    score, category = sess.run([tf.reduce_max(image_net['prob'][0][0][0]), predicted_class],
                               feed_dict={input_tensor:normalised_img[np.newaxis, :, :, :].astype(np.float32), keep_prob: 1.0, is_training: False})
    print('Category:', resnet101_net['meta'][0][0][1][0][0][1][0][category][0])
    print('Score:', score)

    # shape = sess.run(image_net['res5c_relu'], feed_dict={x:normalised_img[np.newaxis, :, :, :].astype(np.float32)}).shape
    # print(shape)

if __name__ == "__main__":
    tf.app.run()
