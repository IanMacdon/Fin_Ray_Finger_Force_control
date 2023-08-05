import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras import datasets
from keras.models import Sequential
from keras import layers
import pathlib


def mask_img(img):
    lower_blue = np.array([50, 180, 45])
    upper_blue = np.array([180, 255, 155])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))
    result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return result


def process_images(all_images):
    output_images = []
    for img in all_images:
        img = cv2.resize(img[:, 200:], shape)
        img = mask_img(img)
        output_images.append(img / 255)
        # cv2.imshow('', img)
        # cv2.waitKey(1)
    return np.asarray(output_images)


objects = []
with (open("normed_finger_data.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

raw_imgs = objects[0]['images']
shape = (400, 400)
images = process_images(raw_imgs)

force = np.asarray(objects[0]['force'])
train_images, test_images, train_labels, test_labels = train_test_split(images, force, train_size=0.8, random_state=42)
print(len(train_images), len(test_images))
print(np.shape(train_images))


model = keras.models.Sequential()
model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(shape[0], shape[1], 1)))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='relu'))

model.summary()

model.compile(optimizer='adam',
             loss=tf.keras.losses.MeanSquaredError(),
             metrics=['mean_squared_error'])

history = model.fit(train_images, train_labels, batch_size=1, epochs=3,
                    validation_data=(test_images, test_labels))

y_pred = model.predict(test_images)
print(np.shape(y_pred))
print(mse(test_labels, y_pred))
plt.figure()
plt.plot(history.history['mean_squared_error'], label='mean_squared_error')
plt.plot(history.history['val_mean_squared_error'], label = 'val_mean_squared_error')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.show()
model.save('cnn7.h5')
