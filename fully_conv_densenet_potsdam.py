from __future__ import division
from __future__ import print_function

from six.moves import xrange

from layers_fc_densenet import *

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_float("learning_rate", "5e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_integer("batch_size", "5", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "../logs-dense/", "path to logs directory")
MAX_ITERATION = int(1e7 + 1)
NUM_OF_CLASSESS = 6
IMAGE_SIZE = 224
tf_records_filename = 'Potsdam.tfrecords'

def inference(image, keep_prob):
    n_filters_first_conv = 48
    n_pool = 5
    growth_rate = 16
    n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    n_classes = 6
    mean_pixel = np.array([97.6398951221, 86.5517502156, 92.5452277039, 85.9159648918, 45.548982716, 31.4374])
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

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'annotation_raw': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.float16)
    annotation = tf.decode_raw(features['annotation_raw'], tf.uint8)
    image = tf.reshape(image, [224, 224, 6])
    annotation = tf.reshape(annotation, [224, 224, 1])
    min_after_deque = 1000
    batch_size = 5
    num_thread = 20
    capacity = min_after_deque + (num_thread + 1) * batch_size
    images, annotations = tf.train.shuffle_batch([image, annotation], batch_size=batch_size, num_threads=num_thread,
                                                 min_after_dequeue=min_after_deque, capacity=capacity)
    return images, annotations

def main(argv=None):
    filename_queue = tf.train.string_input_producer([tf_records_filename])
    image, annotation = read_and_decode(filename_queue)
    image = tf.cast(image, dtype=tf.float32)
    annotation = tf.cast(annotation, dtype=tf.int32)
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")

    pred_annotation, logits = inference(image, keep_probability)
    annotation_64 = tf.cast(annotation, dtype=tf.int64)

    # calculate accuracy for batch.
    cal_acc = tf.equal(pred_annotation, annotation_64)
    cal_acc = tf.cast(cal_acc, dtype=tf.int8)
    acc = tf.count_nonzero(cal_acc) / (FLAGS.batch_size * IMAGE_SIZE * IMAGE_SIZE)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation,
                                                                                            squeeze_dims=[3]),
                                                                          name="entropy")))
    loss_summary = tf.summary.scalar("entropy", loss)
    acc_summary = tf.summary.scalar("accuracy", acc)

    trainable_var = tf.trainable_variables()

    train_op = train(loss, trainable_var)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for itr in xrange(MAX_ITERATION):
        feed_dict = {keep_probability: 0.8}
        sess.run(train_op, feed_dict=feed_dict)
        if itr % 50 == 0:
            train_loss, train_acc, summary_loss, summary_acc = sess.run([loss, acc, loss_summary, acc_summary],
                                                                        feed_dict=feed_dict)
            print("Step: %d, Train_loss: %g, Train_acc: %g" % (itr, train_loss, train_acc))
            train_writer.add_summary(summary_loss, itr)
            train_writer.add_summary(summary_acc, itr)
        if itr % 500 == 0:
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
    coord.request_stop()
    coord.join(threads)

if __name__=='__main__':
    tf.app.run()