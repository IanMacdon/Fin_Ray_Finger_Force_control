import cv2
import numpy as np
import serial
from WSG_Gripper import Gripper
import time
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
import tensorflow as tf


def process_img(img, pos):
    lower_blue = np.array([50, 180, 45])
    upper_blue = np.array([180, 255, 155])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    dst = np.float32([[100, 100], [71, 294], [595, 270]])
    src = np.float32([[162, 280], [140, 480], [462, 353]])
    src[0, 1] = fit1(pos)
    src[1, 1] = fit2(pos)
    src[2, 1] = fit3(pos)
    M = cv2.getAffineTransform(src, dst)
    warpped = cv2.warpAffine(mask, M, (img.shape[1], img.shape[0]))
    kernel = np.ones((5, 5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))
    result = cv2.morphologyEx(warpped, cv2.MORPH_CLOSE, kernel)
    result = cv2.resize(result[:, 200:], (400, 400))
    return warpped


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
    return mask


def read_serial():
    try:
        ser.write(b'1')
        b = ser.readline()
        str_rn = b.decode()
        string = str_rn.rstrip()
        f = float(string)
    except:
        f = 0
    ser.flush()
    return f


y1 = [236, 0]
y2 = [448, 196]
y3 = [335, 195]
pos = [20, 110]
fit1 = np.poly1d(np.polyfit(pos, y1, 1))
fit2 = np.poly1d(np.polyfit(pos, y2, 1))
fit3 = np.poly1d(np.polyfit(pos, y3, 1))


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
# define range of blue color in HSV
lower_blue = np.array([50, 115, 0])
upper_blue = np.array([130, 255, 255])
kernel = np.ones((15, 15), np.uint8)

ser = serial.Serial('COM5', 38400)
wsg = Gripper()
force = []
t = []
start = time.time()
pos = []
print(wsg.homing_gripper())
wsg.homing_gripper()
p = wsg.read_pos()
print(p)
i = 0

images = []
model = load_model('cnn2.h5')
model.summary()
img_force = []
cell_force = []
ret, img = cap.read()
p = wsg.read_pos()
p_img = process_img(img, p)
input_img = p_img[np.newaxis, :, :]
#f = model.predict(input_img, verbose=0)
print(read_serial())
cv2.imshow('', img)
cv2.waitKey(0)
time.sleep(0.5)
wsg.send_speed(2)
s = time.time()
save_img = []
position = []
while p > 35:
    ret, img = cap.read()
    load_f = read_serial()
    p = wsg.read_pos()
    p_img = warp_img(img, p)
    f = read_serial()
    cell_force.append(f)
    save_img.append(p_img)
    position.append(p)


wsg.send_speed(0)
wsg.homing_gripper()
plt.plot(position, cell_force)
plt.show()

objects = []
try:
    with (open("normed_finger_data.pkl", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    save_img = save_img + objects[0]["images"]
    cell_force = cell_force + objects[0]['force']
    position = position + objects[0]['pos']
    dict = {"images": save_img, "force": cell_force,
            "pos": position}
except:
    dict = {"images": save_img, "force": cell_force,
        "pos": position}

plt.plot(cell_force)
plt.show()

with open('normed_finger_data.pkl', 'wb') as f:
    pickle.dump(dict, f)
