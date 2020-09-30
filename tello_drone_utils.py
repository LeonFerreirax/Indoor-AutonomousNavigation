import cv2
from djitellopy import Tello


def initializeTello():
    tccdrone = Tello()
    tccdrone.connect()
    tccdrone.for_back_velocity = 0
    tccdrone.left_right_velocity = 0
    tccdrone.up_down_velocity = 0
    tccdrone.yaw_velocity = 0
    tccdrone.speed = 0
    print(tccdrone.get_battery())
    tccdrone.streamoff()
    tccdrone.streamon()
    return tccdrone

def telloGetFrame(tccdrone, w = 360, h=240):
    mframe = tccdrone.get_frame_read()
    mframe = mframe.frame
    img = cv2.resize(mframe, (w,h))
    return img

