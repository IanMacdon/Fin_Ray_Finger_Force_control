import matplotlib.pyplot as plt
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

ret, img = cap.read()

while True:
    ert, img = cap.read()
    print(ret)
    plt.imshow(img)
    plt.show()
