import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# grab cam, device 0
cap = cv2.VideoCapture(0)

# loop through all frames
while cap.isOpened():
    # read current feed, 2 return values 
    ret, frame = cap.read()

    # show to screen image (not ret)
    cv2.imshow('OpenCV Feed', frame)

    # wait for key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    