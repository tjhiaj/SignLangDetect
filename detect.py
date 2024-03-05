import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split # partition data into training/testing
from tensorflow.keras.utils import to_categorical # convert categorical data via one hot encoding
from tensorflow.keras.models import Sequential # neural network
from tensorflow.keras.layers import LSTM, Dense # temporal component to build network + recognize action, normal full connected layer
from tensorflow.keras.callbacks import TensorBoard # logging
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

# state of the art -> cnn + lstm (low accuracy w/ our data size)
# mp holistic + lstm (less data for accuracy, faster training - dense network - 30-40 mil to .5 mil params)

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

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(112)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63) #so no error thrown when hand out of frame
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

DATA_PATH = os.path.join('MP_Data')                 #path for exported numpy arrays
actions = np.array(['hello', 'thanks', 'iloveyou']) # actions to detect
no_sequences = 30                                   # 30 videos of data
sequence_length = 30                                # each video is 30 frames

# for action in actions:
#     for sequence in range(no_sequences):
#         try:
#             # throws error if directory exists
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass

# cap = cv2.VideoCapture(0) # grab cam, device 0
# # set model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     for action in actions:
#         # loop through videos
#         for sequence in range(no_sequences):
#             for frame_num in range(sequence_length):                     
#                 ret, frame = cap.read()                               # read current feed, 2 return values 
#                 image, results = mediapipe_detection(frame, holistic)
#                 draw_landmarks(image, results)

#                 # wait logic
#                 if frame_num == 0:
#                     cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
#                     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     cv2.waitKey(2000) # 2s break
#                 else:
#                     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

#                 # export keypoints
#                 keypoints = extract_keypoints(results)
#                 npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                 np.save(npy_path, keypoints)

#                 cv2.imshow('OpenCV Feed', image)                      # show to screen image (not ret)
#                 if cv2.waitKey(10) & 0xFF == ord('q'):                # wait for exit key press
#                     break
#     cap.release()
#     cv2.destroyAllWindows()  

# dictionary of action to index
label_map = {label:num for num, label in enumerate(actions)}

# x and y, training model to detect relationship between labels, sequences has 90 videos 30 frames each 1662 keypoints each
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))) # path to npy array
            window.append(res)                                                                        # add frame to window
        sequences.append(window)                                                                      # add video to sequences
        labels.append(label_map[action])

x = np.array(sequences)
y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05) # test partition is 5% of data

# log_dir = os.path.join('Logs')
# tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662))) # num units, next layer needs seq, function, each vid is 30 frames 1662 keypoints
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu')) # fully connected network neurons 
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) # find highest val of vector and predict that action from actions index

# # compile model and feed
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy']) # loss function for multi class classification model, binary is binary_crossentropy, regression uses mean squared error
# model.fit(x_train, y_train, epochs=800, callbacks=[tb_callback]) # small amt of data fits into memory so mp allows you to not build data generator to build pipeline, train on the fly!

# model.summary()
# res = model.predict(x_test)
# actions[np.argmax(res[0])]
# actions[np.argmax(y_test[0])]

# model.save('action.h5')
model.load_weights('action.h5')

# convert one hot encoding to category labels 0,1,2...
yhat = model.predict(x_test)
ytrue = np.argmax(y_test, axis=1).tolist() # axis=1 means want to convert second element in array
yhat = np.argmax (yhat, axis=1).tolist()

# true n, false p, false n, true p
print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))

#new detectin vars for real time predictions
sequence = []   # 30 frames
sentence = []   # concatenate history of detections
threshold = 0.7 # only render results if above this

cap = cv2.VideoCapture(0) # grab cam, device 0
# set model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:  
    while cap.isOpened():                
        ret, frame = cap.read()                               # read current feed, 2 return values 
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:30]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])

            # visualization
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]: #check if current word is different from last
                        sentence.append(actions[np.argmax(res)])
                else:
                    print(actions[np.argmax(res)])
                    sentence.append(actions[np.argmax(res)])
            
            if len(sentence) > 5:
                sentence = sentence[-5:]

        #rendering
        cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1) # -1 means fill
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)                      # show to screen image (not ret)
        if cv2.waitKey(10) & 0xFF == ord('q'):                # wait for exit key press
            break
    cap.release()
    cv2.destroyAllWindows()  