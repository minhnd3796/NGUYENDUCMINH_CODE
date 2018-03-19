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

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
        'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'prob'
    )
    net = {}
    current = image
    for i, name in enumerate(layers):
        if len(name) >= 4:
            kind = name[:4]
        else:
            kind = name[:2]
        if kind == 'conv' or kind == 'fc':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            # kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            kernels = utils.get_variable(kernels, name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            if kind == 'conv':
                current = utils.conv2d_basic(current, kernels, bias)
            elif kind == 'fc':
                current = tf.nn.bias_add(tf.nn.conv2d(current, kernels, strides=[1, 1, 1, 1], padding="VALID"), bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = utils.max_pool_2x2(current)
        elif kind == 'prob':
            current = tf.nn.softmax(current, name=name)
        net[name] = current
    return net

def inference(x, weights):
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, x)
        prediction = tf.argmax(image_net['prob'][0][0][0])

    return prediction, image_net

def main(argv=None):
    img = imread(argv[1])
    # tmp = loadmat('im_.mat')

    vgg19_net = utils.get_model_data('../pretrained_models/imagenet-vgg-verydeep-19.mat')
    mean = vgg19_net['normalization'][0][0][0]
    weights = np.squeeze(vgg19_net['layers'])

    resized_img = resize(img, (224, 224), preserve_range=True, mode='reflect')
    normalised_img = utils.process_image(resized_img, mean)
    
    with tf.device('/cpu:0'):
        x = _input()
        predicted_class, image_net = inference(x, weights)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess.run(tf.global_variables_initializer())
        score, category = sess.run([tf.reduce_max(image_net['prob'][0][0][0]), predicted_class],
                                    feed_dict={x:normalised_img[np.newaxis, :, :, :].astype(np.float32)})
    print('Category:', vgg19_net['classes'][0][0][1][0][category][0])
    print('Score:', score)

if __name__ == "__main__":
    tf.app.run()
