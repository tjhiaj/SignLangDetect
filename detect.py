import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic     #downloads model, makes detections
mp_drawing = mp.solutions.drawing_utils # helps draw keypoints

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # bgr to rgb
    image.flags.writeable = False                  # save memory
    results = model.process(image)                 # predict
    image.flags.writeable = True                   # image now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # rgb to bgr
    return image, results

# render landmarks
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

cap = cv2.VideoCapture(0) # grab cam, device 0
# set model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # loop through all frames
    while cap.isOpened():                      
        ret, frame = cap.read()                               # read current feed, 2 return values 
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        cv2.imshow('OpenCV Feed', image)                      # show to screen image (not ret)
        if cv2.waitKey(10) & 0xFF == ord('q'):                # wait for exit key press
            break
    cap.release()
    cv2.destroyAllWindows()  

# draw_landmarks(frame, results)
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.show()