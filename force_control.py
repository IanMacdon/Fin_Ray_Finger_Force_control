from System import System
from pid import PID
import matplotlib.pyplot as plt
from scipy.signal import lti
from WSG_Gripper import Gripper
from cnn_feedback import ImageFeedback, LoadCallFeedback, MeanFilter
import cv2
import time
import keyboard
import numpy as np


setpoint = 15
dt = 0.06822562217712402
g = lti([33.04, 836.8], [1, 41.08, 809, 0])
G = System(g, dt)
#c = PID(4, 0.45, 0.075, dt)
c = PID(4, 0.25, 0.5, dt)
c.input_min = -400
c.input_max = 400

wsg = Gripper()
force_sensor = ImageFeedback()
load_cell = LoadCallFeedback()
filt = MeanFilter(4)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
pos_vec = []
img_force_vec = []
load_force_vec = []
time_vec = []

ret, img = cap.read()
pos = wsg.read_pos()
img_f = force_sensor.predict_force(img, pos)
wsg.send_speed(0)
start_time = time.time()
last_time = start_time
while True:
    ret, img = cap.read()
    pos = wsg.read_pos()
    force = filt.filter(img_f)
    # u = c.update_controller(setpoint, 110 - pos)
    u = c.update_controller(setpoint, force)
    wsg.send_speed(u)
    pos_vec.append(110 - pos)
    t = time.time()
    img_f = force_sensor.predict_force(img, pos)
    print('Time: ', time.time() - t, 'Force:', img_f)
    img_force_vec.append(force)
    c_time = time.time()
    time_vec.append(c_time - start_time)
    # print('force', force, 'Time', c_time - last_time)
    last_time = c_time
    if keyboard.is_pressed("a"):
        print("You pressed 'a'.")
        break

wsg.homing_gripper()
plt.plot(time_vec, img_force_vec )
plt.show()
