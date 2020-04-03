#some basic imports and setups
import argparse
import datetime
import os
import socket
import threading
import time
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from djitellopy.tello import Tello
import pygame
from alexnet import AlexNet

# standard argparse stuff
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='** = required')
# parser.add_argument('-d', '--distance', type=int, default=3,
#     help='use -d to change the distance of the drone. Range 0-6')
# parser.add_argument('-sx', '--saftey_x', type=int, default=100,
#     help='use -sx to change the saftey bound on the x axis . Range 0-480')
# parser.add_argument('-sy', '--saftey_y', type=int, default=55,
#     help='use -sy to change the saftey bound on the y axis . Range 0-360')
# parser.add_argument('-os', '--override_speed', type=int, default=1,
#     help='use -os to change override speed. Range 0-3')
parser.add_argument('-ss', "--save_session", action='store_true',
    help='add the -ss flag to save your session as an image sequence in the Sessions folder')
parser.add_argument('-D', "--debug", action='store_true',
    help='add the -D flag to enable debug mode. Everything works the same, but no commands will be sent to the drone')

args = parser.parse_args()

#Possíveis comandos
class_names = ["moveForward", "moveLeft", "moveRight", "spinLeft", "spinRight", "stop"]

#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

current_dir = os.getcwd()

#Alterar aqui se preciso o caminho
# diretor = os.path.join(current_dir, 'output/')

# Speed
S = 20
# Frames per second of the pygame window display
FPS = 25

# If we are to save our sessions, we need to make sure the proper directories exist
if args.save_session:
    ddir = "Sessions"

    if not os.path.isdir(ddir):
        os.mkdir(ddir)

    ddir = "Sessions/Session {}".format(str(datetime.datetime.now()).replace(':','-').replace('.','_'))
    os.mkdir(ddir)

class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Create pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with Tello Drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        oSpeed = args.override_speed
        self.speed = 10

        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 50)

    def run(self):

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        # grab the current timestamp and use it to construct the filename
        ts = datetime.datetime.now()

        filename = "-{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))

        #placeholder for input and dropout rate
        x = tf.placeholder(tf.float32, [1, 227, 227, 3])
        keep_prob = tf.placeholder(tf.float32)

        #Create a model with default config( == noskip_layer and 6 unitsin the last layer
        model = AlexNet(x, keep_prob, 6, [])

        #Define activation of last layer as score
        score = model.fc8

        #Create op to calculate softmax
        softmax = tf.nn.softmax(score)

        with tf.Session() as sess:

            #Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Load the pretrained weights into the model
            model.load_initial_weights(sess)

            frame_read = self.tello.get_frame_read()
            imgCount = 0
            should_stop = False

            while not should_stop:
                self.update()

                if frame_read.stopped:
                    frame_read.stop()
                    break

                theTime = str(datetime.datetime.now()).replace(':', '-').replace('.', '_')
                self.screen.fill([0,0,0])
                frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
                frameRet = frame_read.frame

                # Convert image to float32 and resize to (227x227)
                frame = cv2.resize(frame.astype(np.float32), (227,227))

                # Subtract the ImageNet mean
                frame -= imagenet_mean

                # Reshape as needed to feed into model
                frame = frame.reshape((1, 227, 227, 3))
                # frame = np.rot90(frame)
                # frame = np.flipud(frame)
                frame = pygame.surfarray.make_surface(frame)
                self.screen.blit(frame, (0,0))

                # Run the session and calculate the class probability
                self.probs = sess.run(softmax, feed_dict={x: frame,
                                                     keep_prob: 1})

                #Get the class name of the class with the hightest probability
                self.class_name = class_names[np.argmax(self.probs)]

                vid = self.tello.get_video_capture()

                if args.save_session:
                    cv2.imwrite("{}/tellocap{}.jpg".format(ddir, imgCount), frameRet)

                pygame.display.update()

                imgCount += 1

                time.sleep(1 / FPS)

                # Listen for key presses
                k = cv2.waitKey(20)

                # Press T to take off
                if k == ord('t'):
                    if not args.debug:
                        print("Taking Off")
                        self.tello.takeoff()
                        self.tello.get_battery()
                    self.send_rc_control = True

                # Press L to land
                if k == ord('l'):
                    if not args.debug:
                        print("Landing")
                        self.tello.land()
                    self.send_rc_control = False

                if self.send_rc_control:
                    for (forward, left, right, spinLeft, spinRight, stop) in self.probs:
                        if self.class_name == class_names[0]:
                            self.for_back_velocity = int((forward * 10) + S)
                            if left > right:
                                self.yaw_velocity = -int((left * 100) + S)
                            elif left < right:
                                self.yaw_velocity = int((right * 100) + S)
                            else:
                                self.yaw_velocity = 0
                        elif self.class_name == class_names[1]:
                            self.yaw_velocity = -int((left * 10) + S)
                            if (forward >= right):
                                self.for_back_velocity = int((forward * 100) + S)
                            else:
                                self.for_back_velocity = 0
                        elif self.class_name == class_names[2]:
                            self.yaw_velocity = int((right * 10) + S)
                            if (forward >= left):
                                self.for_back_velocity = int((forward * 100) + S)
                            else:
                                self.for_back_velocity = 0
                        elif self.class_name == class_names[3]:
                            self.for_back_velocity = 0
                            self.yaw_velocity = -int((spinLeft * 10) + S)
                            if (forward > left):
                                self.for_back_velocity = int((forward * 100) + S)
                            else:
                                self.for_back_velocity = 0
                                self.yaw_velocity += int((left * 100) + S)
                        elif self.class_name == class_names[4]:
                            self.for_back_velocity = 0
                            self.yaw_velocity = int((spinRight * 10) + S)
                            if (forward > right):
                                self.for_back_velocity = int((forward * 100) + S)
                            else:
                                self.for_back_velocity = 0
                                self.yaw_velocity += int((right * 100))
                        else:
                            if not args.debug:
                                print("Landing")
                                self.tello.land()
                            self.send_rc_control = False


                # Quit the software
                if k == 27:
                    should_stop = True
                    break





                # Display the resulting frame
                cv2.imshow(f'Tello Tracking...', frameRet)

        # for event in pygame.event.get():
        #     if event.type == pygame.USEREVENT + 1:
        #         self.update()
        #     elif event.type == pygame.QUIT:
        #         should_stop = True
        #     elif event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_ESCAPE:
        #             should_stop = True
        #         else:
        #             self.keydown(event.key)
        #     elif event.type == pygame.KEYUP:
        #         self.keyup(event.key)

        # On exit, print the battery
        self.tello.get_battery()

        # When everything done, release the capture
        cv2.destroyAllWindows()

        # Call it always before finishing. To Deallocate resources.
        self.tello.end()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP: # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT: # set backward velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT: # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w: # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s: # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a: # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d: # set yaw clockwise velocity
            self.yaw_velocity = S

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN: # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def battery(self):
        return self.tello.get_battery()[:2]

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                                       self.up_down_velocity, self.yaw_velocity)

def main():


    frontend = FrontEnd()

    # run frontend
    frontend.run()

if __name__ == '__main__':
    main()
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

