import cv2
import numpy as np
import matplotlib.pyplot as plt
from WSG_Gripper import Gripper
import serial
import time
import pickle
import tensorflow as tf
from keras.models import load_model


y1 = [236, 0]
y2 = [448, 196]
y3 = [335, 195]
pos = [20, 110]
fit1 = np.poly1d(np.polyfit(pos, y1, 1))
fit2 = np.poly1d(np.polyfit(pos, y2, 1))
fit3 = np.poly1d(np.polyfit(pos, y3, 1))


def read_serial():
    try:
        ser.write(b'1')
        b = ser.readline()
        str_rn = b.decode()
        string = str_rn.rstrip()
        f = float(string)
    except:
        f = 0
    return f


def warp_img(img, pos):
    dst = np.float32([[100, 100], [71, 294], [595, 270]])
    src = np.float32([[162, 280], [140, 480], [462, 353]])
    src[0, 1] = fit1(pos)
    src[1, 1] = fit2(pos)
    src[2, 1] = fit3(pos)
    M = cv2.getAffineTransform(src, dst)
    warpped = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return warpped


def mask_img(img):
    lower_blue = np.array([50, 180, 45])
    upper_blue = np.array([180, 255, 155])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))
    result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return result / 255


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

ser = serial.Serial('COM5', 38400)
wsg = Gripper()
pos = wsg.read_pos()
model = load_model('cnn4.h5')

ret, img = cap.read()
time.sleep(0.5)
ret, img = cap.read()
cv2.imshow('', img)
cv2.waitKey(0)
p_img = warp_img(img, pos)
p_img = cv2.resize(p_img[:, 200:], (400, 400))
p_img = mask_img(p_img)
input_img = p_img[np.newaxis, :, :]

print(np.shape(input_img))

cv2.imshow('', p_img)
cv2.waitKey(0)
print(model.predict(input_img, verbose=0))
position = []
img_force = []
load_force = []
time_vec = []
s = time.time()
while pos > 40:
    ret, img = cap.read()
    p_img = warp_img(img, pos)
    p_img = cv2.resize(p_img[:, 200:], (400, 400))
    p_img = mask_img(p_img)
    input_img = p_img[np.newaxis, :, :]
    img_f = model.predict(input_img, verbose=0)
    img_force.append(img_f[0][0]-2)
    load_f = read_serial()
    print('g', load_f, 'img force', img_f[0][0])
    load_force.append(load_f)
    pos = wsg.read_pos()
    position.append(110 - pos)
    time_vec.append(time.time() - s)
    # print(img_f[0][0])
wsg.homing_gripper()
plt.plot(time_vec, img_force, time_vec, load_force)
plt.show()
