import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import serial
import tensorflow_model_optimization as tfmot


class ImageFeedback:
    def __init__(self):
        self.model = load_model('cnn4.h5')
        '''
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        tflite_quant_model = converter.convert()
        self.interpreter = tf.lite.Interpreter(tflite_quant_model, num_threads=1)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(type(tflite_quant_model))
        '''
        self.offset = 0
        y1 = [255, 18]
        y2 = [472, 238]
        y3 = [361, 222]
        pos = [25, 110]
        self.fit1 = np.poly1d(np.polyfit(pos, y1, 1))
        self.fit2 = np.poly1d(np.polyfit(pos, y2, 1))
        self.fit3 = np.poly1d(np.polyfit(pos, y3, 1))
        self.lower_blue = np.array([50, 180, 45])
        self.upper_blue = np.array([180, 255, 155])

    def warp_img(self, img, pos):
        dst = np.float32([[100, 100], [71, 294], [595, 270]])
        src = np.float32([[162, 280], [140, 480], [462, 353]])
        src[0, 1] = self.fit1(pos)
        src[1, 1] = self.fit2(pos)
        src[2, 1] = self.fit3(pos)
        M = cv2.getAffineTransform(src, dst)
        warpped = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return warpped

    def mask_img(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))
        result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return result / 255

    def predict_force(self, img, pos):
        p_img = self.warp_img(img, pos)
        p_img = cv2.resize(p_img[:, 200:], (400, 400))
        p_img = self.mask_img(p_img)
        input_img = p_img[np.newaxis, :, :]
        force = self.model.predict(input_img, verbose=0)
        return force[0][0] - self.offset

    def predict_quant_force(self, img, pos):
        p_img = self.warp_img(img, pos)
        p_img = cv2.resize(p_img[:, 200:], (400, 400))
        p_img = self.mask_img(p_img)
        self.interpreter.set_tensor(self.input_details[0]['index'], p_img)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        print(output_data)


class LoadCallFeedback:
    def __init__(self):
        self.ser = serial.Serial('COM5', 38400)

    def read_serial(self):
        try:
            self.ser.write(b'1')
            b = self.ser.readline()
            str_rn = b.decode()
            string = str_rn.rstrip()
            f = float(string)
        except:
            f = 0
        return f


class MeanFilter:
    def __init__(self, size):
        self.history = np.zeros(size)

    def filter(self, force):
        self.history = np.append(force, self.history[0:-1])
        return np.mean(np.asarray(self.history))
