# Vision-based Rubik's Cube Move Detection

<br></br>
**Michael Song**

*University of California, Los Angeles*

## Abstract
This research explores methods for detecting Rubik's Cube rotations from a video stream input. Focusing on three fundamental rotations (right layer counterclockwise, top layer counterclockwise, and front layer counterclockwise denoted as R, U, and F), we divided the task into two main components: identifying frames of movement and classifying sequences of frames into specific moves. To achieve the first task, we propose a two-model system comprising a YOLOv* based object detection model and a classification model. For the second task, we designed three distinct model architectures using related datasets: an LSTM-based model, a Convolutional LSTM, and a combination of two CNNs. In our limited dataset experiments, the LSTM and ConvLSTM both achieved over 80% accuracy, while the BiCNN achieved over 40%. Furthermore, a combined model incorporating both the LSTM and ConvLSTM improved on both and achieved over 90% accuracy.

## 1. Introduction
This study draws inspiration from the emergence of smart cubes like the Giker cube, which can record cube rotations, times, scrambles, and more. These cubes, however, have faced criticism for their subpar feel compared to flagship speed cubes from brands like GAN, Moyu, and Qiyi. This research explores the potential of computer vision techniques to replace smart cubes, allowing cubers to record and replay their solves on their preferred speed cubes. Additionally, it seeks to improve on modern video based classification, particularly movement recognition.

## 2. Related Work
While Rubik's Cube-specific research is limited, numerous papers have addressed computer vision tasks involving sequences of frames. Our model architectures draw on previous works, including Karen Simonya and Andrew Zisserman's "Two-Stream Convolutional Neural Networks for Action Recognition in Videos" [1] and Shi et al.'s "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting." [2]

## 3. Methodology 
The pipeline of this project can be separated into two main sections. The first is responsible
for processing the input stream (video) and separating the video into sequences of frames
that are moves. The second makes inferences based on these image folders, and the output is
one of three moves R, U, and F. The next sections detail the steps and processes behind each
component.

### 3.1 Input Stream Processing
As mentioned earlier, the input stream processing will convert the input, in this case 
video, into sequences of frames that will be used for inference. At the core of this section
is two computer vision models: an object detection model and a classification model. The
object detection model is responsible for locating the Rubik's Cube in every frame, 
identifying the region of interest used for both the classification model and in some of the
move prediction models. On the other hand, the classification model is used to predict whether the Rubik's Cube is in motion. 
This is important because we ideally only want to run the move prediction models on sequences
of frames that we are most likely sure to contain moves, which avoids having to infer on
every sequence of ten frames in the video. To train these models, we used Roboflow to help process our datasets,
which were hand recorded videos of example Rubik's Cube turns. Through the Roboflow interface,
we were able to easily annotate over 200 images, which were automatically split into training,
 validation, and testing data. Additionally, Roboflow supplemented the original 200 labeled
data by performing data augmentation, including a horizontal flip, vertical flip, and -15
to 15 degrees rotation. Afterward, we fine-tuned a pretrained YOLOv8 object detection model
on the newly created dataset using the YOLOv8 API. The classification model was trained in a similar
method, with the only differences being that we used the earlier object detection model as a
preprocessing step, which locates the region of interest in each frame. This way, we isolated
the Rubik's Cube from the background image and allowed the classification model to solely 
predict based on the pixels from the cube. By using the YOLOv8 API to train these models,
we were also able to record the accuracy and loss during training, and obtain useful metrics
that can help us improve performance in the future. We will discuss the results of these
models in a later section.

With both models fully trained, we separate the input stream into frames using the OpenCV
library, then crop each frame using the object detection model, and predict whether the cube
is in motion using the classification model. Furthermore, we also store the state of the previous
frame, such that we can check if the state of motion has changed. If the cube has changed
from stationary to moving, we then create a sequence folder and begin adding subsequent frames
until we identify that the cube is no longer in motion. As we want to make move predictions
on the fly, we then immediately pass that sequence of frames into the model used for move
prediction. 

### 3.2 Move Prediction
The move prediction model is responsible for taking a sequence of frames, and making a prediction
about what move is being shown in the sequence. We consider three potential model architectures
here, and the datasets behind each one. There is an overarching dataset that is shared by 
all subsequent datasets, which each make their own modifications to it. This dataset is comprised
of a "sequences" folder, containing folders "sequence1", "sequence2", etc. Additionally, there
is a "metadata.csv" file that helps with loading the data. The frames in each sequence were
handpicked from a plethora of recorded videos using the same background and similar location, all
which only include the three types of rotations specified earlier. 
In the following sections, we will examine these different architectures more closely.

#### 3.2.1 LSTM Model
The first model we propose is an LSTM model. Compared to the other models, this model is relatively
simple, and is implemented with a single LSTM cell and a fully connected layer. Compared to
a standard RNN, this model is better at retaining information across a series of time, and
thus chosen for this task. The primary focus of this model is to make a move prediction based
on hand landmarks during a rotation. To obtain landmark information from each frame, we
use MediaPipe's Hand Landmark Task model, which accurately locates 21 landmarks per hand, totaling
to 42 landmarks across two hands. Each landmark has three values, an x, y, z coordinate, and thus
the sample input data for a given frame is a (42, 3) array. Due to the nature of a LSTM model,
we can pass in a variable length sequence in the forward pass, and this results in an input
data of size (N, sequence_length, 42, 3), where N represents the number of sequences in the
batch, and sequence_length is the number of frames in that sequence. Finally, there is a 
fully connected layer that takes inputs from the last layer of the LSTM and converts that
data into a three element output array. One issue that results with this architecture is 
that in some frames, the MediaPipe model isn't able to detect a hand, and in those cases, 
we decided to fill in the array with dummy values of 0. In the future, we would like to come up
with better methods to handle these edge cases.

#### 3.2.2 ConvLSTM Model
The second model that we examine is a Convolutional LSTM. Using the Convolutional LSTM
model from Shi et al. implemented by Andrea Palazzi and Davide Abati. On top of the standard
ConvLSTM cell from the paper, we also add a linear layer and a softmax layer for prediction. 
Contrary to the first model, this model operates on a cropped rubik's cube dataset. The cropping
is done by the object detection model presented earlier, and the training dataset still uses
the same videos and frames, with the only difference being that the rubik's cube is cropped
frome each individual frame. After testing a few different configurations, we settled on a
architecture consisting of a 64 sized hidden layer, 3x3 convolution kernel, and 3 total layers. 

#### 3.2.3 BiCNN Model
The last model we present is an architecture consisting of two Convolutional Neural Networks,
each serving a different purpose. The inspiration for this model was from Simonyan and 
Zisserman's "Two-Stream Convolutional Networks for Action Recognition in Videos", who used
this style of architecture to perform video classification. Because videos are spatiotemporal
data, the idea is to process the spacial data and temporal data separately. In this manner,
the first CNN of this system is used on spacial information, static images of Rubik's Cube
rotations. In Simonyan and Zisserman's application, they noted that simply with the spatial
CNN alone, they were able to achieve a high level of accuracy in video classification, as
many actions were drastically different from one another. However, since our application
only deals with slight differences between move rotations, the spatial model performed
similarly to that of a randomized prediction model. To reinforce this first model, we introduce
a second CNN to help with temporal information. For our case, we use the optical flow
between frames and concatenate that flow over a series of frames as data. As an example, 
say that we had a sequence of 10 frames. Then, we can obtain 9 optical flow frames, and each
of these 9 optical flows would have 2 dimensions: an x displacement and y displacement. To
help with calculating the flow fields, we use OpenCV's calcOpticalFlowFarneback and then
transform the data by stacking the x displacements on top of the y displacements. Additionally,
since the input to the CNN must be consistent across all frames, we decide to randomly sample
a sequence of 6 frames within a rotation, leading to a optical flow array of (5x224x224x2), which
we reshape to (10x224x224). The specific architectures of each model are detailed in 
"Two-Stream Convolutional Networks for Action Recognition in Videos", and are very similar
to the AlexNet CNN model used in Image Detection models back in the days. 

## 4. Results

Before looking at the results of the models, here are some metrics from the YOLOv8 models along
with example validation batches.

**YOLOv8 Object Detection Model**


![image](/runs/detect/train23/confusion_matrix_normalized.png)
![image](/runs/detect/train23/F1_curve.png)
![image](/runs/detect/train23/PR_curve.png)

**YOLOv8 Classification Model**


![image](/runs/classify/train4/confusion_matrix_normalized.png)
![image](/runs/classify/train4/val_batch0_pred.jpg)


| Model Type          | Test Accuracy<br/>(Avg over 10 tests) |
|---------------------|---------------------------------------|
| **LSTM**            | 82.35%                                |
| **ConvLSTM**        | 91.84%                                |
| **BiCNN**           | 41.42%                                |
| **LSTM + ConvLSTM** | 94.12%                                |

We tested many different model parameters and hyperparameters throughout the project, 
and we found interesting results regarding the structure and performance. Starting with
the finger landmark LSTM model, our model performed best with the following configurations:
            input_dim=3,
            hidden_dim=128,
            output_dim=3,
            num_landmarks=42,
            num_layers=3.
We noticed that when we increased the complexity of this model by increasing the number of layers,
the model was unable to learn accurately, and we also see a similar trend in the ConvLSTM. We
hypothesize that this is due to the limited size of the training data. With such a small
training dataset, the model isn't able to accurately adjust all the weights and biases;
it essentially has too many settings to tune and not enough knowledge to know which setting
does what. The accuracy displayed are results averaged over 10 tests, with a testing data of around
40 sequences. Another interesting observation with the LSTM is that the current model does
not include a softmax layer to normalize the output. When we added a softmax layer after
the fully connected layer, we documented an inability to actually learn the data, resulting
in accuracies between 20 and 30 percent, which is equivalent to a randomized guessing algorithm.


Our ConvLSTM model showed slightly better results than our finger landmark based LSTM. Over
10 tests, our trained ConvLSTM reached an accuracy of 91.84% on our custom testing dataset. 
Similar to the LSTM, when we increased the number of layers to 5 and added more hidden
dimensions, the model failed to train well. For example, when we defined our ConvLSTM
as having input_dim=3, hidden_dim=64, kernel_size=(7,7), num_layers=3, we achieved
a test accuracy of 34.6%. Instead, our best results with the ConvLSTM
was with input_dim=3, hidden_dim=32, kernel_size=(3,3), num_layers=1, resulting in
an accuracy of 89.80% after 200 epochs. We noticed consistent learning but oscillations
of loss between epochs, which lead to us using an adaptive learning rate of alpha=0.0001
after 200 epochs. Using this scaled learning rate produced the accuracy shown in the 
table of 91.84%. 


The BiCNN model performed as expected given the previous results, and with the model
architecture defined in "Two-Stream Convolutional Networks for Action Recognition in Videos"
we achieved poor accuracy ranging between 30 and 40%. We opted to train each model
separately, as was done in the paper, but our results weren't great. Again, this is 
likely due to the small sample size not being able to accurately train the large amount
of parameters in the BiCNN. Reducing the complexity of both CNN's will likely increase
performance by a substantial margin, but was not tested in this project.

While not mentioned previously, we also examined a fourth model that combines both
the LSTM and ConvLSTM by applying a fully connected layer on top of the last layers
of both models. The goal of this model is to utilize the benefits of the LSTM and ConvLSTM
to produce results that reach even higher accuracies. Throughout testing, we documented
that this combined model had an average of 94.12% accuracy, which indeed is the highest of the
model tested.


## 5. Conclusion
In this paper, we proposed a end to end custom pipeline for detecting and classifying Rubik's Cube moves. 
To handle the input stream, we utilized two YOLOv8 based object detection and classification
models that identified the region of interest in each frame, and determined the state of motion
of the cube. For this project, we focused our attention on the latter half of the 
pipeline; training three different prediction models to classify move sequences. 
The first was a finger landmark based LSTM model, which achieved a decent level
of accuracy at 80%. The second was a Convolutional LSTM model that made predictions on
the region of interest in each frame and improved on the original model, obtaining an 
accuracy of over 85%. Noting the success of these two models, we furthermore trained a
model that combined the LSTM and ConvLSTM, resulting in the best accuracy of 94%. Our
third prediction model was a combination of two CNN's that operated separately on spatial
and temporal data. While the performance of the last model didn't match the earlier models,
it can likely also obtain good results with more time to modify and experiment. Throughout,
our training session, we noticed that due to the significantly small size of training
sequences, our models failed to learn appropriately when overly complex. Contrastingly,
models with simple architectures such as having only one layer and a fewer number of hidden
dimensions tended to drastically outperform it's complex counterparts. 

Moving forward, we believe there is much research potential in generalizing the entire move
detection pipeline. The models trained in this project were highly fitted towards
a specific type of turning style, Rubik's Cube size (3x3), and camera angle. With 
a much larger training dataset and better architecture, we propose that it is very likely
to achieve amazingly accurate move detection with low latency that matches that of bluetooth
technology. 

## Works Cited

1. Simonyan, K., Zisserman, A. "Two-Stream Convolutional Neural Networks for Action Recognition in Videos." *Journal of Computer Vision*, 2014.
2. Shi, X., Chen, Z., Wang, H., Yeung, D., Wong W., Woo W. "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" *NeurIPS*, 2015. 
