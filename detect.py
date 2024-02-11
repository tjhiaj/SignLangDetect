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
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1), 
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

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

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(112)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63) #so no error thrown when hand out of frame
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

print(extract_keypoints(results)).shape