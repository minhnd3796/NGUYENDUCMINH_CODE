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

def construct_batch_normalisation_block(current, net, weights, weight_start_index, param_names, layer_names):
    scale = weights[weight_start_index + 1][1].reshape(-1)
    scale = utils.get_variable(scale, name=param_names[1])

    offset = weights[weight_start_index + 2][1].reshape(-1)
    offset = utils.get_variable(offset, name=param_names[2])

    mean = weights[weight_start_index + 3][1][:, 0].reshape(-1)
    mean = utils.get_variable(mean, name=param_names[3] + '_mean')

    variance = weights[weight_start_index + 3][1][:, 1].reshape(-1)
    variance = utils.get_variable(variance, name=param_names[3] + '_variance')
    variance_epsilon = 1e-5

    current = tf.add(tf.multiply(scale,
        tf.divide(tf.subtract(current, mean),
            tf.sqrt(tf.add(tf.square(variance), variance_epsilon)))), offset, name=layer_names[1])
    net[layer_names[1]] = current

    return current, net


def construct_conv1_block(current, net, weights, weight_start_index, param_names, layer_names):
    # conv1
    conv1_filter = weights[weight_start_index][1]
    kernel_size = conv1_filter.shape[0]
    padding = int((kernel_size - 1) / 2)
    conv1_filter = utils.get_variable(conv1_filter, name=param_names[0])
    current = tf.nn.conv2d(
        tf.pad(current, tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]])),
        conv1_filter, strides=[1, 2, 2, 1],
        padding="VALID",
        name=layer_names[0])
    net[layer_names[0]] = current

    # bn_conv1
    current, net = construct_batch_normalisation_block(current, net, weights, weight_start_index, param_names, layer_names)

    # conv1_relu
    current = tf.nn.relu(current, name=layer_names[2])
    net[layer_names[2]] = current

    # pool1
    current = tf.nn.max_pool(current, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=layer_names[3])
    net[layer_names[3]] = current

    return current, net

def create_branch1_layer_names(res_num):
    name_list = [None] * 2
    name_list[0] = 'res' + str(res_num) + 'a' + '_branch1'
    name_list[1] = 'bn' + str(res_num) + 'a' + '_branch1'
    return name_list

def create_param_names_from_layers(layer_names):
    param_list = [None] * 4
    param_list[0] = layer_names[0] + '_filter'
    param_list[1] = layer_names[1] + '_mult'
    param_list[2] = layer_names[1] + '_bias'
    param_list[3] = layer_names[1] + '_moments'
    return param_list

def construct_branch1_block(current, net, weights, weight_start_index, branch1_param_names, branch1_layer_names):
    return None, None

def resnet101_net(weights, image):
    net = {}
    current = image

    # conv1 block
    conv1_layer_names = ['conv1', 'bn_conv1', 'conv1_relu', 'pool1']
    conv1_param_names = create_param_names_from_layers(conv1_layer_names)
    current, net = construct_conv1_block(current, net, weights, 0, conv1_param_names, conv1_layer_names)
    pool1 = current

    # res2 block
    branch1_layer_names = create_branch1_layer_names(2)
    branch1_param_names = create_param_names_from_layers(branch1_layer_names)
    print(branch1_layer_names)
    print(branch1_param_names)
    # current, net = construct_branch1_block(current, net, weights, 4, branch1_param_names, branch1_layer_names)
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
    
    with tf.device('/cpu:0'):
        x = _input()
        image_net = inference(x, weights)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        # score, category = sess.run([tf.reduce_max(image_net['prob'][0][0][0]), predicted_class],
        #                             feed_dict={x:normalised_img[np.newaxis, :, :, :].astype(np.float32)})
        inspected_values = sess.run(image_net['pool1'], feed_dict={x:tmp['im_'][np.newaxis, :, :, :].astype(np.float32)})
    print(np.sum(inspected_values))
    print(inspected_values.shape)
    # print('Category:', resnet101_net['classes'][0][0][1][0][category][0])
    # print('Score:', score)

if __name__ == "__main__":
    tf.app.run()
