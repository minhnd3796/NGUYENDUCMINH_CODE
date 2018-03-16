import tensorflow as tf

tf_records_filename = 'Vaihingen.tfrecords'

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
    image = tf.reshape(image, [224, 224, 15])
    annotation = tf.reshape(annotation, [224, 224, 1])
    min_after_deque = 10
    batch_size = 5
    num_thread = 16
    capacity = min_after_deque + (num_thread + 1) * batch_size
    images, annotations = tf.train.shuffle_batch([image, annotation], batch_size=batch_size, num_threads=num_thread,
                                                 min_after_dequeue=min_after_deque, capacity=capacity)
    return images, annotations


def demo():
    filename_queue = tf.train.string_input_producer([tf_records_filename])
    image, annotation = read_and_decode(filename_queue)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord= tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(2):
            img, anno = sess.run([image, annotation])
            print(img.shape)

            print('current batch')

            # We selected the batch size of two
            # So we should get two image pairs in each batch
            # Let's make sure it is random

            # imshow(img[0, :, :,0:3])
        coord.request_stop()
        coord.join(threads)
if __name__=='__main__':
    demo()