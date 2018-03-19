from sys import argv
import tensor_utils as utils
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from scipy.misc import imread
from skimage.transform import resize

def _input():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    return x

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

def construct_test_batch_normalisation_block(current, net, weights, weight_start_index, param_names, layer_names):
    scale = weights[weight_start_index + 1][1].reshape(-1)
    scale = utils.get_variable(scale, name=param_names[1])

    offset = weights[weight_start_index + 2][1].reshape(-1)
    offset = utils.get_variable(offset, name=param_names[2])

    mean = weights[weight_start_index + 3][1][:, 0].reshape(-1)
    mean = utils.get_variable(mean, name=param_names[3] + '_mean')

    variance = weights[weight_start_index + 3][1][:, 1].reshape(-1)
    variance = utils.get_variable(variance, name=param_names[3] + '_variance')

    current = tf.add(tf.multiply(scale, tf.divide(tf.subtract(current, mean), variance)), offset, name=layer_names[1])

    net[layer_names[1]] = current

    return current, net

def construct_conv_bn_block(current, net, weights, weight_start_index, param_names, layer_names, stride):
    # conv
    kernels = weights[weight_start_index][1]
    kernels = utils.get_variable(kernels, name=param_names[0])
    if stride == 1:
        current = tf.nn.conv2d(current, kernels, strides=[1, 1, 1, 1], padding='SAME', name=layer_names[0])
    else:
        kernel_size = kernels.shape[0]
        padding = int((int(kernel_size) - 1) / 2)
        current = tf.nn.conv2d(
            tf.pad(current, tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]])),
            kernels, strides=[1, stride, stride, 1], padding='VALID', name=layer_names[0])
    net[layer_names[0]] = current

    # bn
    print(layer_names)
    print(param_names)
    print()
    current, net = construct_test_batch_normalisation_block(current, net, weights, weight_start_index, param_names, layer_names)
    return current, net

def construct_conv_bn_relu_block(current, net, weights, weight_start_index, param_names, layer_names, stride):
    # conv_bn
    current, net = construct_conv_bn_block(current, net, weights, weight_start_index, param_names, layer_names, stride)
    # relu
    current = tf.nn.relu(current, name=layer_names[2])
    net[layer_names[2]] = current
    return current, net

def resnet101_net(weights, image):
    net = {}
    current = image
    param_start_index = 0

    # conv1 block
    conv1_layer_names = ['conv1', 'bn_conv1', 'conv1_relu', 'pool1']
    conv1_param_names = create_param_names_from_layers(conv1_layer_names)
    current, net = construct_conv_bn_relu_block(current, net, weights, param_start_index, conv1_param_names, conv1_layer_names, 2)
    current = tf.nn.max_pool(current, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=conv1_layer_names[3])
    net[conv1_layer_names[3]] = current
    param_start_index += 4

    # res2a_branch1 block
    branch1_layer_names = create_branch_layer_names(res_num=2, res_type='a', branch_type=1)
    branch1_param_names = create_param_names_from_layers(branch1_layer_names)
    current, net = construct_conv_bn_block(net['pool1'], net, weights, param_start_index, branch1_param_names, branch1_layer_names, 1)
    param_start_index += 4

    # res2a_branch2 block
    branch2a_layer_names = create_branch_layer_names(res_num=2, res_type='a', branch_type=2, branch_order='a')
    branch2a_param_names = create_param_names_from_layers(branch2a_layer_names)
    current, net = construct_conv_bn_relu_block(net['pool1'], net, weights, param_start_index, branch2a_param_names, branch2a_layer_names, 1)
    return net

def inference(x, weights):
    with tf.variable_scope("inference"):
        image_net = resnet101_net(weights, x)
        # prediction = tf.argmax(image_net['prob'][0][0][0])

    # return prediction, image_net
    return image_net

def main(argv=None):
    
    tmp = loadmat('MatConvNet Tutorials/im_dagnn_tank1.mat')

    resnet101_net = utils.get_model_data('../pretrained_models/imagenet-resnet-101-dag.mat')
    weights = np.squeeze(resnet101_net['params'])

    """ img = imread(argv[1])
    mean = resnet101_net['meta'][0][0][2][0][0][2]
    resized_img = resize(img, (224, 224), preserve_range=True, mode='reflect')
    normalised_img = utils.process_image(resized_img, mean) """
    
    x = _input()
    image_net = inference(x, weights)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # score, category = sess.run([tf.reduce_max(image_net['prob'][0][0][0]), predicted_class],
    #                             feed_dict={x:normalised_img[np.newaxis, :, :, :].astype(np.float32)})
    inspected_values = sess.run(image_net['res2a_branch2a_relu'], feed_dict={x:tmp['im_'][np.newaxis, :, :, :].astype(np.float32)})
    print(np.sum(inspected_values))
    print(inspected_values.shape)
    # print('Category:', resnet101_net['classes'][0][0][1][0][category][0])
    # print('Score:', score)

if __name__ == "__main__":
    tf.app.run()
