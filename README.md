
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

> The os library lets me use OS dependent funcionality and the os.path module specifically lets me manipulate paths. os.path.join() concatenates the path-like objects passed to it. It adds directory separators after every nonempty element! In my case, I pass the single argument 'MP_Data' so that is my final path.

Then, I create a NumPy array that stores my actions 'hello', 'thanks', and 'iloveyou' [line 51]

I also specify how many videos I will be capturing for each action (30) and how many frames each video will have (30) [lines 52 & 53]

Now that I know I want 30 videos for each action, I need to make these subdirectories to store the data. So in 'MP_Data' I want subfolders 'hello', 'thanks', and 'iloveyou'. In each of these action folders, I want subfolders numbered 0 to 29, representing each of the videos. [lines 55-61]

---
Great! My directories are all set up. Now, I need to capture my training data. To start off, I grab my camera aka device 0 [line 63]

> cv2 is the main module in OpenCV that offers image and video processing functions. cv2.VideoCapture() opens the camera at the index/id provided. In my case, I open the default device 0.

Next, I use the MediaPipe Holistic Model to start collecting landmark data [line 65]

> The MediaPipe Holistic pipeline integrates separate models for pose, face and hand components. This allows live perception of simultaneous human pose, face landmarks, and hand tracking in real-time.

I loop through my actions and their respective videos and, for each video, I capture its frames one by one using the VideoCapture.read() method from cv2 [line 70] Note that this method returns a tuple (return value, image) where the return value is checked for a successful reading before using the image.

With each frame, I want to make a detection to capture the landmark values. To do this, I need to convert my current image from its BGR colour code to RGB for my model to interpret it properly [line 21] I also need to set my image to Read Only before any processing so that it'll be passed by reference, saving memory [line 22] Then, I use the MediaPipe Holistic Model to make a detection. [line 23] I want to return this detection along with the image in its original state, so I make the image writable and convert the colour back to BGR before returning [lines 24-26]

Now, I have the original image and the detection for it. Next, I want to render the landmarks on the image. For each part (face, pose, left hand, right hand), I use the draw_landmarks method from MediaPipe. I pass to it the image, the x,y,z landmarks of the part, and the drawing specifications for the dots and lines representing the keypoints and connections between them [lines 29-41]

Then, I need some logic to indicate where I am in the collection process AND pause between videos so I can reset. Every time I start collecting for a video, I display 'STARTING COLLECTION' onscreen along with the action name and video number. Then, I wait 2 seconds before proceeding [lines 74-78] Remaining frames for that video don't get a pause (only the first one does!) [lines 79-80]

> cv2.putText() is a method used to draw text strings on any image. Here, I pass to it the image, text, x-y position, font, font size, colour, line thickness, and line type. We use LINE_AA (anti-aliased) for a smoother look since pixels are added along the edges to blend with the background.

Next, I want to export my keypoints. For each part (face, pose, left hand, right hand), I create a numpy array that stores an array for each landmark [lines 43-47] These nested arrays will contain the xyz coordinates of the landmark. Then, I flatten everything into one big array. Note that this only happens if the landmark collection of the part is nonempty. Otherwise, an array of zeros is assigned to that part. Finally, I return a giant array with the arrays of all parts concatenated together [line 48]

> I know exactly how many zeros I need based on how many landmarks are tracked by the model for that part. For example, the pose model tracks 33 landmarks and each landmark gets 4 array entries (x, y, z, visbility). Thus, I need a zero-array with 33*4=132 entries.
>
> Note: only pose has a visibility variable

Then, I want to store the array of keypoints from that frame in its respective folder (recall, we made these directories at the start). I create the path MP_Data/action/video_num/frame_num and store the array in that folder [lines 82-85]

Finally, I display the image onscreen using cv2.imshow(window_name, image) and wait 10ms to see if the exit key (q) is pressed [lines 87-88] If a key is pressed, cv2.waitKey() returns a corresponding integer. Note that 0xFF is 11111111 in binary and the integer returned by cv2.waitKey() also has a binary representation. We use bitwise AND (&) which means each bit in 0xFF is compared to each bit in the returned integer. If both bits are 1, the resulting bit is 1. Otherwise, the resulting bit is 0. This gives us the last 8 bits of the returned integer and masks all previous bits to 0. We check if these 8 bits match the ASCII code of 'q' and, if so, break the loop [line 89] Different platforms will return different integers for pressed keys but the last 8 bits usually match the ASCII codes cross-platform. Once out of the loop, the camera is released and all windows are destroyed [lines 90-91]

---
Yay, that's all my training data! Now, I create a dictionary that maps action to index using Python's eunmerate() [line 94] enumerate() takes any object that supports iteration (aka an iterable, in our case the actions array) and returns a sequence of tuples, each containing an index and a value from the iterable. Now I can access any action in this dictionary and get its associated index, so I can refer to the action by its index.

Next, I create 2 empty arrays: sequences and labels [line 97] Sequences will contain 90 videos (30 for each action), each video having 30 frames and each frame having 1662 keypoints (sum of all landmarks tracked by the models for each part - face, left hand, right hand, pose). Labels will store the index of the action.

Then, I set up the loops to populate these arrays. For each frame, I load the data from the numpy array stored in MP_Data from our previous training data collection. The String format() method allows me to replace the braces with the respective frame number [lines 98-105] I store the data from these frames inside a window array that represents a video. Once all the data is appended for a window, the big window array gets appended to the Sequences array. The index of the action (0, 1, or 2) also gets appended to the Labels array. This repeats for all 90 videos (30 videos for each of the 3 actions).

Great! My Sequences and Labels arrays are populated. Now, I turn Sequences into a numpy array so it's a valid input type for the train_test_split() method [line 107] I do the same with the Labels array using to_categorical(). to_categorical() creates a binary matrix representation of Labels. For example, if my Labels array is [0, 1, 2] then to_categorical will give me [[1. 0. 0.], [0. 1. 0.], [0. 0. 1.]]. I cast these values back to integers using pandas astype() [line 108] This process is called one-hot encoding and it's necessary because the actions 'hello', 'thanks', and 'iloveyou' have no particular order. Integer encoding them as 0, 1, 2 and passing these values to the model as if order matters may result in poor performance or unexpected results, such as values between 0 and 1 or 1 and 2 (for example 0.5 is an unwanted value we may get that is not associated with any action). 

Next, I split my arrays into random train and test subsets [line 110] The training data is used to teach the model while the test set is used after training to evaluate the final model. It's important to have a test set because it serves as "new" data that the model hasn't seen before and thus will help me validate its performance. The test_size is the proportion included in the test split. In my case, 5% of the data is included in the test partition while 95% is in the train partition.

---

Now comes the fun part: prepping the model! I start off by loading/instantiating the Sequential neural network from TensorFlow. Sequential is a good fit because my model will be a plain stack of layers, each layer having one input and one output tensor. A tensor is just an n-dimensional matrix. In my case, during training, the input tensor would have size (85, 30, 1662) where 85 videos are being passed, each made up of 30 frames that each comprise of 1662 keypoint values. The first layer will output a tensor of shape (85, 30, 64) since return_sequences=True means all hidden states are returned, each comprising of 64 units. Every frame/timestep will output a hidden state (usually used to store a prediction) and these hidden states are all returned when return_sequences=True. The last LSTM layer has return_sequences set to False so that the output has size (85, 64) since only the last element of the sequence is returned. This acts as a summary of the last 30 timesteps.

A unit is simply a cell that contains a memory state and a gate mechanism that controls the flow of information. You'll see that different layers have different numbers of units (64, 128, etc.) This number is usually a power of 2 but is determined through experimentation. The goal is to reach the sweet spot of complexity that yields the best results. We want to track enough features that we can attain expected results but not too many that it overcomplicates the learning process and confuses the model. 

You'll also notice we use an activation function. Activation functions calculate the output of a node based on its inputs and their weights. They allow us to introduce non-linear relationships. In my case, I'm using relu or ReLU which stands for rectified linear unit and has form f(x) = max(0, x). This means it is linear for all inputs greater than 0, but returns 0 for all negative inputs. It's the most popular function because it is almost linear (easy to optimize), mitigates the vanishing gradient problem, computes efficiently (negative inputs set to 0 mean only some nodes are activated), and is scalabe (simple threshold at 0). 

> The vanishing gradient problem is a phenomenon where gradients become vanishingly small during backpropagation. A gradient is the change in weights with respect to the change in error or, as Fridman says, the change in ouput wen you change the input (kind of like the slope of a line). In my case, using relu, the gradient is preserved as 1 for all positive inputs since the function is linear for positive inputs. 
>
> Backpropagation is an algorithm to calculate the gradient (change in network's weights) to reduce the cost/error of the calculation. It starts at the final layer and looks at the difference between the current and desired outputs. Many factors can be changed to achieve this desired output including, weights, activation, and biases. The network looks at how it can change the weights/biases in that layer BUT activations can't be changed directly, so what it does is move to the previous layer and look at how changing the weights/biases there can give the desired activation change in the following layer. It repeats this process as it moves backwards and eventually sums all these little weight/bias changes between the layers until we have a change value for each node. This happens for every training example so that there are change values for every node for every training example. Then, all the change values for each node are averaged which helps find the gradient.
>
> The problem is that when weights and activations are small (less than 1), their products get smaller and smaller and so does the gradient. Tiny gradients mean the network's weights change less and less, eventually having no impact on optimizing the network. Relu prevents this by preserving the gradient at 1.

In the final layer, I use the softmax activation function. This function simplifies the network output to a vector of probabilities for each category. In my case, it outputs a vector of size 3 consisting of the probabilities for each action.

It's also important to note the difference between LSTM and Dense layers. A Long short-term memory (LSTM) layer is designed to handle input that's an ordered sequence, where information from earlier in the sequence can help with the prediction. LSTM layers are a type of Recurrent Neural Network (RNN) layer, meaning they use the ouput from a previous step along with the next element in the sequence as input for the next step. LSTM nodes have states that serve as a sort of "memory" to store information, and gates that determine which information to forget/recall and how mjuch the ouput should be affected by saved information versus the current calculation.

Dense layers are fully-connected layers, meaning every neuron in the layer is connected to every neuron in the next layer, connecting each input to each output. Their primary purpose is to change the dimension of the data. In my case, I want to take the 64-dimensional vector from the last LSTM layer and make it 3-dimensional to determine which action was detected. Dense layers are great for multi class classification since the number of nodes equals the number of outputs. That's why I use it in the last layer, with 3 nodes for 3 outputs, 1 for each action.

---

The next step is to compile the model. I use the compile() method that configures the model for training. The optimizer is an algorithm that changes the weights of a network during training to optimize it (minimize its loss function). Here, I uses 'Adam' (a combo of optimizers AdaGrad and RMSProp) which is known for its efficiency and stability. It requires little memory, making it useful for deep learning models with many parameters, such as LSTM networks. The algorithm performs gradient calculation, moment estimates, bias correction, and a parameter update. Gradients were discussed earlier in this text. Moments are used to smooth/stabilize the gradient updates. The first moment is the exponential moving average (EMA) of the gradients, capturing the average direction of the gradients. The second moment is the EMA of the squared gradients, capturing the average magnitude of the gradients.

> EMA is a moving average that places greater weight on more recent data points than older ones.

Bias correction is needed because the moment estimates are initialized to 0 and so are biased towards 0. Finally, the parameters are updated based on the final moment estimates.

A loss function tells me how well my algorithm models my dataset. It will output a high number if my predictions aren't accurate, and a low one if they are. categorical_crossentropy is the loss function used in multi-class classification models where there are many categories and the model must decide which one the input belongs to. It allows the model to output the probabilities of each category, allowing me to evaluate its confidence if needed.

Finally, the list of metrics to be evaluated by the model when training/testing. In my case, categorical_accuracy is the standard metric to assess the performance of a multi-class classification model on categorical label data. This calculation represents how often the prediction matches the actual "true" one-hot labels. It checks if the index of the max true value is equal to the index of the max predicted value.

A loss function tells me how well my algorithm models my dataset. It will output a high number if my predictions aren't accurate, and a low one if they are. categorical_crossentropy is the loss function used in multi-class classification models where there are many categories and the model must decide which one the input belongs to. It allows the model to output the probabilities of each category, allowing me to evaluate its confidence if needed.

Finally, the list of metrics to be evaluated by the model when training/testing. In my case, categorical_accuracy is the standard metric to assess the performance of a multi-class classification model on categorical label data. This calculation represents how often the prediction matches the actual "true" one-hot labels. It checks if the index of the max true value is equal to the index of the max predicted value.

Then, I train the model using the fit() method. It takes in input data (my sequences), target data (my labels, known to be correct), the number of epochs (iterations over the input and target data), and the list of callbacks to apply during training. Callbacks are objects that can perform actions at various stages of training. tb_callback [line 113] enables visualizations for TensorBoard. TensorBoard() takes in the the path of the directory where the log files to be parsed by TensorBoard will be saved. In my case, I create and use a Logs folder. After training, I save my model as action.h5 [line 132] so I can load all those carefully calculated post-training weights whenever needed using the load_weights() method [line 133]

---

Now, I start setting up for my real time predictions. I set up my sequence array similar to the one I set up for training. I also create a sentences array to help me concatenate the history of detections (previously detected actions) in my display. Then, I have the threshold where I set the minimum confidence level for results to be rendered.

As before, I grab my camera and set the holistic model. I use a while loop to ensure the program runs continuously while the camera is on. The camera reading, mediapipe detection, landmark drawing, and keypoint extraction is identical to the steps we did in training. Then, I insert the keypoints at the start of my sequence array and trim the array so it only includes the last 30 items (sets of keypoints) that were added.

As long as I have enought keypoint data, I'll make a prediction using the predict() method. First, I use the numpy expand_dims() method to expand the shape of the array and add an axis at position 0. Initially, sequence has shape (30, 1662) but adding an axis gives shape (1, 30, 1662). This reshaping is necessary because the model expects input of shape (samples, time steps, features). It expects the samples dimension to be the batch size so we reshape it with a batch size of 1 sample. The output of predict() has shape (1, 3) where the only element of the array is an array of size 3 with the probabilities of each action, so we take that element as our result.

Next comes visualization. If the probability of the predicted action exceeds the set threshold, the "sentence" of actions displayed already has words in it, and the predicted action is different from the last one, I append it to the sentence to be displayed. This prevents the display of action history from being populated entirely by one action, since predictions occur quite frequently. If the probability of the predicted action exceeds the set threshold but there are no existing words in the sentence, I append it to the sentence without checking for previous words. I also only display the last 5 actions at any given time.

prob_viz() is a function that visualizes the probabilities of each action using coloured bars that dynamically represent how likely it is that the model believes that action is being displayed. It takes in a result (array of probabilities), the array of possible actions, an input frame, and the colours used in the visualization. First, it enumerates the result, adding a counter to the iterable. In my case, result is an array of length 3 with the probabilities of each action. enumerate() returns an enumerate object of the form [(0, prob_1), (1, prob_2), (2, prob_3)]. prob_viz loops through these number-probability pairs and draws the repective rectangle on a copy of the input frame. cv2.rectangle() takes in the frame, start point coordinates, end point coordinates, colour in bgr, and rectangle thickness in px. prob_viz also displays the appropriate action name for each rectangle using cv2.putText() which takes in the output frame, action name, coordinates of bottom-left of text box, font scale (multiplier for base size), colour in bgr, thickness in px, and line type.

Once I get an image from prob_viz(), I have a couple more elements to add. First, I add the bar that will display the detection history. Using cv2.rectangle(), I draw a blue bar at the top of the screen. Then, I add the text (the sentence of the last 5 detected actions). The rest resembles the loop we created at the start to collect training data: showing the image to the screen, waiting for the exit key to be pressed, releasing the camera, and closing the window.

All done! Thanks for following along :)
