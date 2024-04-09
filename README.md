
# Sign Language Detection

A sign language model that integrates keypoint extraction, LSTM-based action detection modeling, and real-time prediction from video sequences. Applies TensorFlow and Keras to construct a deep neural network.


## Features

- Renders probabilities of each phrase in real time
- Displays last 5 predictions
- Refined keypoint rendering


## Screenshots

![App Screenshot](/sign-lang.png)


## Tech Stack

**Language:** Python

**Frameworks:** OpenCV, Tensorflow, Keras

**Dev Tools:** NumPy, Matplotlib


## How I Built It

First, I specify the path to the folder 'MP_Data' where I will store my training data [line 50]

    The os library lets me use OS dependent funcionality and the os.path module specifically lets me manipulate paths. os.path.join() concatenates the path-like objects passed to it. It adds directory separators after every nonempty element! In my case, I pass the single argument 'MP_Data' so that is my final path.

Then, I create a NumPy array that stores my actions 'hello', 'thanks', and 'iloveyou' [line 51]

I also specify how many videos I will be capturing for each action (30) and how many frames each video will have (30) [lines 52 & 53]

Now that I know I want 30 videos for each action, I need to make these subdirectories to store the data. So in 'MP_Data' I want subfolders 'hello', 'thanks', and 'iloveyou'. In each of these action folders, I want subfolders numbered 0 to 29, representing each of the videos. [lines 55-61]

---
Great! My directories are all set up. Now, I need to capture my training data. To start off, I grab my camera aka device 0 [line 63]

    cv2 is the main module in OpenCV that offers image and video processing functions. cv2.VideoCapture() opens the camera at the index/id provided. In my case, I open the default device 0.

Next, I use the MediaPipe Holistic Model to start collecting landmark data [line 65]

    The MediaPipe Holistic pipeline integrates separate models for pose, face and hand components. This allows live perception of simultaneous human pose, face landmarks, and hand tracking in real-time.

I loop through my actions and their respective videos and, for each video, I capture its frames one by one using the VideoCapture.read() method from cv2 [line 70] Note that this method returns a tuple (return value, image) where the return value is checked for a successful reading before using the image.

With each frame, I want to make a prediction. To do this, I need to convert my current image from its BGR colour code to RGB for my model to interpret it properly [line 21] I also need to set my image to Read Only before any processing so that it'll be passed by reference, saving memory [line 22] Then, I use the MediaPipe Holistic Model to make a prediction. [line 23] I want to return this prediction along with the image in its original state, so I make the image writable and convert the colour back to BGR before returning [lines 24-26]
