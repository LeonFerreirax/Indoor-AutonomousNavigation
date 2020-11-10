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
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from alexnet import AlexNet
from tensorflow.python.tools import inspect_checkpoint as chkp
seed_value = 1234
tf.keras.backend.clear_session()
np.random.seed(seed_value)
tf.set_random_seed(seed_value)
random.seed(seed_value)

#def reset_session(seed):
#  tf.keras.backend.clear_session()
#  tf.compat.v1.set_random_seed(seed)
#  np.random.seed(seed)
#  random.seed(seed)
#reset_session(seed=seed_value)

batch_size = 1
#Possíveis comandos
class_names = ["moveForward", "moveLeft", "moveRight",
               "spinLeft", "spinRight", "stop"]



def test_image(path_image, num_class, weights_path='Default'):
    #tf.set_random_seed(seed=seed_value)
    tf.reset_default_graph()

    keep_prob = tf.placeholder(dtype=tf.float32, shape=[],name="keep_prob")
    #
    img_string = tf.read_file(path_image)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    print(type(img_decoded))
    # img_decoded = tf.image.decode_jpeg(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_resized = tf.reshape(img_resized, shape=[batch_size, 227, 227, 3])

    # AlexNet
    saver = tf.train.import_meta_graph('D:/Docs/tcc_indoor_final/finetune_alexnet/checkpoint/model_epoch178.ckpt.meta')
    model = AlexNet(img_resized, keep_prob, 6, skip_layer='', weights_path=saver)
    score = tf.nn.softmax(model.fc8)
    max = tf.arg_max(score, 1)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver.restore(sess,
                      "D:/Docs/tcc_indoor_final/finetune_alexnet/checkpoint/model_epoch178.ckpt")
        # score = model.fc8
        print(sess.run(model.fc8, feed_dict={keep_prob: 1.0}))
        pred = sess.run(score, feed_dict={keep_prob: 1.0})
        print("pred:", pred)
        prob = sess.run(max, feed_dict={keep_prob: 1.0})[0]
        print(prob)

        class_name = class_names[np.argmax(pred)]
        print(class_name)

        for (forward, left, right, spinLeft, spinRight, stop) in pred:
             print(forward)
             soma = int(forward * 100) + 5
             print(soma)
             print(left)
             esquerda = int(left * 10) + 5
             print(esquerda)
             print(right)
             print(spinLeft)
             print(spinRight)
             print(stop)

        # matplotlib
        plt.imshow(img_decoded.eval())
        plt.title("Class:" + class_names[prob] + ",probability: %.4f" % pred[0, np.argmax(pred)])
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
 
# class_names = ["moveForward", "moveLeft", "moveRight", "spinLeft", "spinRight", "stop"]
#
# #mean of imagenet dataset in BGR
# imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
#
# current_dir = os.getcwd()
# image_dir = os.path.join(current_dir, '3_Corridor/exp432')
#
# # get list of all images
# img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
#
# def encode(frame, ovstream, output):
#     """
#     convert frames to packets and write to file
#     """
#     try:
#         pkt = ovstream.encode(frame)
#     except Exception as err:
#         print("encoding failed{0}".format(err))
#
#     if pkt is not None:
#         try:
#             output.mux(pkt)
#         except Exception:
#             print('mux failed: '), str(pkt)
#     return True
#
# # load all images
# imgs = []
# for f in img_files:
#     imgs.append(cv2.imread(f))

# plot images
# fig = plt.figure(figsize=(15, 6))
# for i, img in enumerate(imgs):
#     fig.add_subplot()
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()

#placeholder for input and dropout rate
# x = tf.placeholder(tf.float32, [1, 227, 227, 3])
# keep_prob = tf.placeholder(tf.float32)
#
# #create model with default config ( == no skip_layer and 1000 units in the last layer)
# model = AlexNet(x, keep_prob, 6, [])
#
# #define activation of last layer as score
# score = model.fc8
#
# #create op to calculate softmax
# softmax = tf.nn.softmax(score)
#
# with tf.Session() as sess:
#     # Initialize all variables
#     sess.run(tf.global_variables_initializer())
#
#     # Load the pretrained weights into the model
#     model.load_initial_weights(sess)
#
#     # Create figure handle
#     fig2 = plt.figure(figsize=(15, 6))
#
#     # Loop over all images
#     for i, image in enumerate(imgs):
#         # Convert image to float32 and resize to (227x227)
#         img = cv2.resize(image.astype(np.float32), (227, 227))
#
#         # Subtract the ImageNet mean
#         img -= imagenet_mean
#
#         # Reshape as needed to feed into model
#         img = img.reshape((1, 227, 227, 3))
#
#         # Run the session and calculate the class probability
#         probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
#
#         # Get the class name of the class with the highest probability
#         class_name = class_names[np.argmax(probs)]
#
#         # Plot image with class name and prob in the title
#         fig2.add_subplot()
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.title("Class: " + class_name + ", probability: %.4f" % probs[0, np.argmax(probs)])
#         plt.axis('off')
#         plt.show()

