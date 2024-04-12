
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

First, I grab the model I'll be using to make detections [line 17] and the module containing functions that'll help me draw keypoints [line 18] Then, I specify the path to the folder 'MP_Data' where I will store my training data [line 50]

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

With each frame, I want to make a detection to capture the landmark values. To do this, I need to convert my current image from its BGR colour code to RGB for my model to interpret it properly [line 21] I also need to set my image to Read Only before any processing so that it'll be passed by reference, saving memory [line 22] Then, I use the MediaPipe Holistic Model to make a detection. [line 23] I want to return this detection along with the image in its original state, so I make the image writable and convert the colour back to BGR before returning [lines 24-26]

Now, I have the original image and the detection for it. Next, I want to render the landmarks on the image. For each part (face, pose, left hand, right hand), I use the draw_landmarks method from MediaPipe. I pass to it the image, the x,y,z landmarks of the part, and the drawing specifications for the dots and lines representing the keypoints and connections between them [lines 29-41]

Then, I need some logic to indicate where I am in the collection process AND pause between videos so I can reset. Every time I start collecting for a video, I display 'STARTING COLLECTION' onscreen along with the action name and video number. Then, I wait 2 seconds before proceeding [lines 74-78] Remaining frames for that video don't get a pause (only the first one does!) [lines 79-80]

    cv2.putText() is a method used to draw text strings on any image. Here, I pass to it the image, text, x-y position, font, font size, colour, line thickness, and line type. We use LINE_AA (anti-aliased) for a smoother look since pixels are added along the edges to blend with the background.

Next, I want to export my keypoints. For each part (face, pose, left hand, right hand), I create a numpy array that stores an array for each landmark [lines 43-47] These nested arrays will contain the xyz coordinates of the landmark. Then, I flatten everything into one big array. Note that this only happens if the landmark collection of the part is nonempty. Otherwise, an array of zeros is assigned to that part. Finally, I return a giant array with the arrays of all parts concatenated together [line 48]

    I know exactly how many zeros I need based on how many landmarks are tracked by the model for that part. For example, the pose model tracks 33 landmarks and each landmark gets 4 array entries (x, y, z, visbility). Thus, I need a zero-array with 33*4=132 entries.
    Note: only pose has a visibility variable

Then, I want to store the array of keypoints from that frame in its respective folder (recall, we made these directories at the start). I create the path MP_Data/action/video_num/frame_num and store the array in that folder [lines 82-85]

Finally, I display the image onscreen using cv2.imshow(window_name, image) and wait 10ms to see if the exit key (q) is pressed [lines 87-88] If a key is pressed, cv2.waitKey() returns a corresponding integer. Note that 0xFF is 11111111 in binary and the integer returned by cv2.waitKey() also has a binary representation. We use bitwise AND (&) which means each bit in 0xFF is compared to each bit in the returned integer. If both bits are 1, the resulting bit is 1. Otherwise, the resulting bit is 0. This gives us the last 8 bits of the returned integer and masks all previous bits to 0. We check if these 8 bits match the ASCII code of 'q' and, if so, break the loop [line 89] Different platforms will return different integers for pressed keys but the last 8 bits usually match the ASCII codes cross-platform. Once out of the loop, the camera is released and all windows are destroyed [lines 90-91]
