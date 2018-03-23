from __future__ import print_function

from scipy.misc import imread, imsave

import tensor_utils_5_channels as utils
from layers_fc_densenet import *

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "../ISPRS_semantic_labeling_Vaihingen", "path to dataset")
MAX_ITERATION = int(1e6 + 1)
NUM_OF_CLASSESS = 6
IMAGE_SIZE = 224

def inference(image, keep_prob):
    n_filters_first_conv = 48
    n_pool = 4
    growth_rate = 12
    n_layers_per_block = [5]*11
    n_classes = 6
    mean_pixel = np.array([120.895239985, 81.9300816234, 81.2898876188, 66.8837693324, 30.6986130799, 283.307])
    processed_image = utils.process_image(image, mean_pixel)
    print(np.shape(processed_image))
    W_first = utils.weight_variable([3,3,processed_image.get_shape().as_list()[3],n_filters_first_conv], name='W_first')
    b_first = utils.bias_variable([n_filters_first_conv], name= 'b_first')
    conv_first = utils.conv2d_basic(processed_image, W_first, b_first)
    stack = tf.nn.relu(conv_first)
    n_filters = n_filters_first_conv
    print("Before Downsample")
    print(np.shape(stack))
    #####################
    # Downsampling path #
    #####################

    skip_connection_list = []
    for i in range(n_pool):
        # Dense Block
        for j in range(n_layers_per_block[i]):
            l = BN_ReLU_Conv(inputs=stack,n_filters= growth_rate,keep_prob=keep_prob, name="downsample_"+str(i)+"_"+str(j))
            stack = tf.concat([stack,l], axis=3)
            n_filters += growth_rate
        skip_connection_list.append(stack)
        stack = Transition_Down(inputs=stack, n_filters=n_filters, keep_prob=keep_prob, name='downsample_stack_'+str(i))

    skip_connection_list = skip_connection_list[::-1]

    #####################
    #     Bottleneck    #
    #####################
    block_to_upsample = []
    for j in range(n_layers_per_block[n_pool]):
        l = BN_ReLU_Conv(inputs=stack, n_filters=growth_rate, keep_prob= keep_prob, name="bottleneck_"+str(j))
        block_to_upsample.append(l)
        stack = tf.concat([stack,l], axis=3)

    #######################
    #   Upsampling path   #
    #######################

    for i in range(n_pool):
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = Transition_Up(skip_connection=skip_connection_list[i], block_to_upsample=block_to_upsample, n_filters_keep = n_filters_keep, name="upsample_stack_"+str(i))

        # Dense Block
        block_to_upsample = []
        for j in range(n_layers_per_block[n_pool + i + 1]):
            l = BN_ReLU_Conv(inputs=stack, n_filters=growth_rate, keep_prob=keep_prob, name="upsample_"+str(i)+"_"+str(j))
            block_to_upsample.append(l)
            stack = tf.concat([stack, l], axis=3)

    W_last = utils.weight_variable([1,1,stack.get_shape().as_list()[3],n_classes], name="W_last")
    b_last = utils.bias_variable([n_classes], name="b_last")
    conv_last = utils.conv2d_basic(stack,W_last,b_last)
    print("Conv_last")
    print(np.shape(conv_last))
    annotation_pred = tf.argmax(conv_last, dimension=3, name="prediction")
    return tf.expand_dims(annotation_pred, dim=3), conv_last


def infer_little_img(input_image_path,patch_size=224,stride_ver=112,stride_hor=112):
    tf.reset_default_graph()
    input_image= imread(input_image_path)
    dsm_image= imread(input_image_path.replace('top','dsm').replace('_mosaic','').replace('area','matching_area'))
    dsm_image = np.expand_dims(dsm_image, axis=2)
    ndsm_image= imread(input_image_path.replace('top/','ndsm/').replace('top','dsm').replace('_mosaic','')
                       .replace('area','matching_area').replace('.tif','_normalized.jpg'))
    ndsm_image= np.expand_dims(ndsm_image,axis=2)
    ndvi_image = imread(input_image_path.replace('top', 'ndvi').replace('_mosaic_09cm_area', ''))
    ndvi_image = np.expand_dims(ndvi_image, axis=2)
    height = np.shape(input_image)[0]
    width = np.shape(input_image)[1]
    output_image = np.zeros_like(input_image)
    input_image= np.concatenate((input_image, ndvi_image,ndsm_image,dsm_image),axis=2)
    output_map = np.zeros((height, width, 6), dtype=np.float32)
    number_of_vertical_points = (height - patch_size) // stride_ver + 1
    number_of_horizontial_points = (width - patch_size) // stride_hor + 1
    sess= tf.Session()
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, 6], name="input_image")
    _, logits = inference(image, keep_probability)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    input_image= np.expand_dims(input_image,axis=0)
    print(np.shape(input_image))
    for i in range(number_of_vertical_points):
        for j in range(number_of_horizontial_points):
            current_patch = input_image[:,i * stride_ver:i * stride_ver + patch_size,
                            j * stride_hor:j * stride_hor + patch_size, :]
            logits_result = sess.run(logits, feed_dict={image: current_patch, keep_probability: 1.0})
            logits_result = tf.squeeze(logits_result)
            patch_result= sess.run(logits_result)
            output_map[i * stride_ver:i * stride_ver + patch_size, j * stride_hor:j * stride_hor + patch_size,
            :] += patch_result
            print('stage 1: i='+str(i)+"; j="+str(j))
    for i in range(number_of_vertical_points):
        current_patch= input_image[:,i*stride_ver:i*stride_ver+patch_size,width-patch_size:width,:]
        logits_result = sess.run(logits, feed_dict={image: current_patch, keep_probability: 1.0})
        logits_result = tf.squeeze(logits_result)
        patch_result = sess.run(logits_result)
        output_map[i*stride_ver:i*stride_ver+patch_size,width-patch_size:width,:]+=patch_result
        print('stage 2: i=' + str(i) + "; j=" + str(j))
    for i in range(number_of_horizontial_points):
        current_patch= input_image[:,height-patch_size:height,i*stride_hor:i*stride_hor+patch_size,:]
        logits_result = sess.run(logits, feed_dict={image: current_patch, keep_probability: 1.0})
        logits_result = tf.squeeze(logits_result)
        patch_result = sess.run(logits_result)
        output_map[height-patch_size:height,i*stride_hor:i*stride_hor+patch_size,:]+=patch_result
        print('stage 3: i=' + str(i) + "; j=" + str(j))
    current_patch = input_image[:,height - patch_size:height, width - patch_size:width, :]
    logits_result = sess.run(logits, feed_dict={image: current_patch, keep_probability: 1.0})
    logits_result = tf.squeeze(logits_result)
    patch_result = sess.run(logits_result)
    output_map[height - patch_size:height, width - patch_size:width, :] += patch_result
    predict_annotation_image = np.argmax(output_map, axis=2)
    print(np.shape(predict_annotation_image))
    for i in range(height):
        for j in range(width):
            if predict_annotation_image[i,j]==0:
                output_image[i,j,:]=[255,255,255]
            elif predict_annotation_image[i,j]==1:
                output_image[i,j,:]=[0,0,255]
            elif predict_annotation_image[i,j]==2:
                output_image[i,j,:]=[0,255,255]
            elif predict_annotation_image[i,j]==3:
                output_image[i,j,:]=[0,255,0]
            elif predict_annotation_image[i,j]==4:
                output_image[i,j,:]=[255,255,0]
            elif predict_annotation_image[i,j]==5:
                output_image[i,j,:]=[255,0,0]
    return output_image

if __name__ == "__main__":
    #tf.app.run()
    imsave("top_mosaic_09cm_area35.tif",
           infer_little_img("/home/khmt/sangdv/duytv/Thesis_15channels/ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area35.tif"))