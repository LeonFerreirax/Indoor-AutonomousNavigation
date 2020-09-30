import tensorflow as tf
import numpy as np
import sys
from alexnet import AlexNet


# Edit just these
FILE_PATH = tf.train.latest_checkpoint('D:/Docs/indoor/finetune_alexnet/checkpoints')
NUM_CLASSES = 6
OUTPUT_FILE = 'D:/Docs/indoor/tf_alexnet/alexnet_epoch1.npy'


if __name__ == '__main__':
    # placeholder for input and dropout rate
    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)
    model = AlexNet(x, keep_prob, 6, [])

    saver = tf.train.Saver()
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    data = {
        'conv1': [],
        'conv2': [],
        'conv3': [],
        'conv4': [],
        'conv5': [],
        'fc6': [],
        'fc7': [],
        'fc8': []
    }

    with tf.Session() as sess:
        saver.restore(sess, FILE_PATH)

        for op_name in layers:
          with tf.variable_scope(op_name, reuse = True):
            biases_variable = tf.get_variable('biases')
            weights_variable = tf.get_variable('weights')
            data[op_name].append(sess.run(biases_variable))
            data[op_name].append(sess.run(weights_variable))

        np.save(OUTPUT_FILE, data)