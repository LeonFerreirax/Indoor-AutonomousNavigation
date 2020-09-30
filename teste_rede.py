#Basic imports and setups
import argparse
import datetime
import os
import random
import socket
import threading
import time
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from alexnet import AlexNet

def reset_session(seed):
  tf.keras.backend.clear_session()
  tf.compat.v1.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
reset_session(seed=0)

#batch_size
batch_size = 1
#Possíveis comandos
class_names = ["moveForward", "moveLeft", "moveRight",
               "spinLeft", "spinRight", "stop"]
metagrap_path='D:/Docs/tcc_indoor_final/finetune_alexnet/checkpoint/model_epoch178.ckpt.meta'
checkpoint_path = 'D:/Docs/tcc_indoor_final/finetune_alexnet/checkpoint/model_epoch178.ckpt'

def test_image(path_image, num_class, weights_path='Default'):
    tf.reset_default_graph()

    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')

    img_string = tf.read_file(path_image)
    img_decoded = tf.image.decode_jpeg(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227,227])
    img_resized = tf.reshape(img_resized, shape=[batch_size, 227, 227, 3])

    #Alexnet
    model = AlexNet(img_resized, keep_prob, 6, skip_layer='', weights_path=weights_path)
    score = tf.nn.softmax(model.fc8)
    max = tf.arg_max(score, 1)

    #Saver
    saver = tf.train.import_meta_graph(metagrap_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_path)

        print(sess.run(model.fc8, feed_dict={keep_prob: 1.0}))
        pred = sess.run(score, feed_dict={keep_prob: 1.0})
        print(pred)
        prob = sess.run(max, feed_dict={keep_prob: 1.0})[0]
        print(prob)
        class_name = class_names[np.argmax(pred)]
        print(class_name)

        for (forward, left, right, spinLeft, spinRight, stop) in pred:
            print('Forward: ', forward)
            print('Left: ', left)
            print('Right: ', right)
            print('spinLeft: ', spinLeft)
            print('spinRight: ', spinRight)
            print('Stop: ', stop)

        plt.imshow(img_decoded.eval())
        plt.title("Comando:" + class_names[class_name] + ",probability: %.4f" % pred[0, np.argmax(pred)])
        plt.axis('off')
        plt.show()


image_path = 'D:/Docs/tcc_indoor/Indoor-AutonomousNavigation/4teste'
data = []

for filename_1 in os.listdir(os.path.join(image_path)):
    print(filename_1)
    filename_path = image_path+'/'+filename_1
    data.append(filename_path)
for j in data:
    print(j)
    test_image(j, num_class=6)