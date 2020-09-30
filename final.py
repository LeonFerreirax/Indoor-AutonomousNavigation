import random
from djitellopy import Tello
import cv2
import time
import datetime
import os
import numpy as np
import argparse
import tensorflow as tf
from alexnet import AlexNet

def reset_session(seed):
  tf.keras.backend.clear_session()
  tf.compat.v1.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
reset_session(seed=0)

#Argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='** = required')
parser.add_argument('-ss', "--save_session", action='store_true',
    help='add the -ss flag to save your session as an image sequence in the Sessions folder')
parser.add_argument('-D', "--debug", action='store_true',
    help='add the -D flag to enable debug mode. Everything works the same, but no commands will be sent to the drone')

args = parser.parse_args()

class_names = ["moveForward", "moveLeft", "moveRight", "spinLeft", "spinRight", "stop"]
#Alterar caminho aqui
metagrap_path = './tf_alexnet/model_epoch178.ckpt.meta'
#E aqui
checkpoint_path = './tf_alexnet/model_epoch178.ckpt'

if args.save_session:
    ddir = "Sessions"
    if not os.path.isdir(ddir):
        os.mkdir(ddir)
    ddir = "Sessions/Session {}".format(str(datetime.datetime.now()).replace(':', '-').replace('.', '_'))
    os.mkdir(ddir)

width = 320
height = 240
ts = datetime.datetime.now()
filename = "-{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))

#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

tf.reset_default_graph()

imgCount = 0
startCounter = 0

S = 20

#Inicialize Tello
mdrone = Tello()
mdrone.connect()
mdrone.for_back_velocity = 0
mdrone.left_right_velocity = 0
mdrone.up_down_velocity = 0
mdrone.yaw_velocity = 0
mdrone.speed = 0
mdrone.send_rc_control = False
should_stop = False
print(mdrone.get_battery())
mdrone.streamoff()
mdrone.streamon()


while True:

    frame_read = mdrone.get_frame_read()
    mframe = frame_read.frame
    img = cv2.resize(mframe, (width, height))

    inp = cv2.resize(img, (227, 227))
    inp = inp[:, :, [2, 1, 0]] # BGR2RGB

    img_reshape = inp.reshape(inp, shape=[1, 227, 227, 3])

    #Alexnet
    model = AlexNet(img_reshape, keep_prob, 6, skip_layer='', weights_path='Default')
    score = tf.nn.softmax(model.fc8)
    max = tf.arg_max(score, 1)

    #Saver
    saver = tf.train.import_meta_graph(metagrap_path)

    with tf.Session() as sess:
        #Initialize all variables
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_path)

        # Calculo da probabilidade
        print('Matriz: ', sess.run(model.fc8, feed_dict={keep_prob: 1.0}))
        prediction = sess.run(score, feed_dict={keep_prob: 1.0})
        print("Prediction: ", prediction)
        prob = sess.run(max, feed_dict={keep_prob: 1.0})[0]
        print(prob)
        class_name = class_names[np.argmax(prediction)]
        print('Categoria: ', class_name)

        vid = mdrone.get_video_capture()

        if args.save_session:
            cv2.imwrite("{}/tellocap{}.jpg".format(ddir, imgCount), mframe)

        #Listen to anu key press
        k = cv2.waitKey(20)

        #Press t to take off
        if k == ord('t'):
            if not args.debug:
                print("Taking off")
                mdrone.takeoff()
                mdrone.get_battery()
                mdrone.send_rc_control = True
        #Press L to land
        if k == ord('l'):
            if not args.debug:
                print("Landing")
                mdrone.land()
                mdrone.send_rc_control = False

        if mdrone.send_rc_control:
            mdrone.send_rc_control(mdrone.left_right_velocity, mdrone.for_back_velocity,
                                   mdrone.up_down_velocity, mdrone.yaw_velocity)
            for (forward, left, right, spinLeft, spinRight, stop) in prediction:
                if prediction == class_names[0]:
                    print("Frente")
                    mdrone.for_back_velocity = int((forward * 10) + S)
                    if left > right:
                        mdrone.yaw_velocity = -int((left * 100) + S)
                    elif left < right:
                        mdrone.yaw_velocity = int((right * 100) + S)
                    else:
                        mdrone.yaw_velocity = 0
                elif prediction == class_names[1]:
                    print("Esquerda")
                    mdrone.yaw_velocity = -int((left * 10) + S)
                    mdrone.for_back_velocity = int((forward * 100) + S)
                    if (forward < right):
                        mdrone.for_back_velocity = 0
                elif prediction == class_names[2]:
                    print("Direita")
                    mdrone.yaw_velocity = int((right * 10) + S)
                    mdrone.for_back_velocity = int((forward * 100) + S)
                    if (forward < left):
                        mdrone.for_back_velocity = 0
                elif prediction == class_names[3]:
                    print("Giro a esquerda")
                    mdrone.for_back_velocity = 0
                    mdrone.yaw_velocity = -int((spinLeft * 10) + S)
                    if (forward > left):
                        mdrone.for_back_velocity = int((forward * 100) + S)
                    else:
                        mdrone.yaw_velocity += int((left * 100) + S)
                elif prediction == class_names[4]:
                    print("Giro a Direita")
                    mdrone.for_back_velocity = 0
                    mdrone.yaw_velocity = int ((spinRight * 10) + S)
                    if (forward > right):
                        mdrone.for_back_velocity = int((forward * 100) + S)
                    else:
                        mdrone.yaw_velocity += int((right * 100) + S)
                elif prediction == class_names[5]:
                    print("Parada")
                    mdrone.for_back_velocity = 0
                    mdrone.yaw_velocity = 0
                    mdrone.rotate_clockwise(90)
                    mdrone.land()
                    mdrone.send_rc_control = False

            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(mframe,'Comando:' + class_names[class_name] + ",probability: %.4f" % prediction[0, np.argmax(prediction)])
            cv2.imshow("demo", cv2.cvtColor(mframe, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)
            time.sleep(3)

        # WAIT FOR THE 'Q' BUTTON TO STOP
        if cv2.waitKey(1) or k == ord('q'):
            mdrone.land()
            cv2.destroyAllWindows()
            mdrone.end()
            break
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, res, (int(img.shape[0] / 3), int(img.shape[1] / 3)), font, 1, (0, 0, 255),
        #            2)  # putting on the labels
        #cv2.imshow("demo", img)
        #cv2.waitKey(0)