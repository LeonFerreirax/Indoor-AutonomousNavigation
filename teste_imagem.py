import tensorflow as tf
from alexnet import AlexNet
import numpy as np
import cv2
import matplotlib.pyplot as plt

keep_prob = 0.5
num_classes = 6
skip_layer = []
batch_size = 1
#Possíveis comandos
class_names = ["moveForward", "moveLeft", "moveRight",
               "spinLeft", "spinRight", "stop"]
# Test image reading path
image = cv2.imread("./4teste/23.jpg")
img_resized = cv2.resize(image, (227, 227))

x = tf.placeholder(tf.float32, [1, 227, 227, 3], name='x-input')
# Define neural network structure, the model initialization
saver = tf.train.import_meta_graph('D:/Docs/tcc_indoor_final/finetune_alexnet/checkpoint/model_epoch178.ckpt.meta')
model = AlexNet(x, keep_prob, num_classes, skip_layer, weights_path=saver)
score = model.fc8
#softmax output layer neural network to obtain front propagating
softmax = tf.nn.softmax(score)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,
                  "D:/Docs/tcc_indoor_final/finetune_alexnet/checkpoint/model_epoch178.ckpt")

    test = np.reshape(img_resized, (1, 227, 227, 3))
    # Sess.run () function returns the run tensor is the corresponding array
    soft = sess.run([softmax], feed_dict={x: test})
    # Gets the index where the maximum is located
    maxx = np.argmax(soft)
    print(maxx)
    # Find the target belongs to the category
    class_name = class_names[np.argmax(soft)]
    print(class_name)
    text = 'Predicted class:' + str(maxx) + '(' + class_name + ')'
    # Display test categories
    cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the probability of that category
    cv2.putText(image, 'with probability:' + str(soft[0][maxx]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('test_image', image)
    # For 10 seconds
    cv2.waitKey(10000)