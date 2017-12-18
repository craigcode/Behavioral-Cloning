# **Behavioral Cloning** 

## Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I took the advice offered in the course and borrowed from the
Nvidia Model Pipeline referenced here: https://arxiv.org/pdf/1604.07316v1.pdf


#### 2. Attempts to reduce overfitting in the model

I introduced a Dropout layer with 0.5 probability to handle overfitting.

#### 3. Model parameter tuning

I compiled the model with an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I encountered some problems keeping the car on the track during early stage training,
which I attributed to driving the car like a game trying to hold the racing line.
Keeping things centered proved far more fruitful for successful autonomous track navigation
later. 



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used an implementation of the suggested Nvidia Model Pipeline.

See https://arxiv.org/pdf/1604.07316v1.pdf

Normalization and cropping provided a better usable window of the image upfront.
I then mirrored the Nvidia convolution layers and fully-connected layers in Keras
condensing to a single output and compiled with an Adam optimizer.

To prevent overfitting, I later included a dropout layer.


#### 2. Final Model Architecture


| Layer (type)                    | Output Shape        | Param  | Connected to          |
| ------------------------------- |:-------------------:| ------:|:---------------------:|
| lambda_1 (Lambda)               | (None, 160, 320, 3) | 0      | lambda_input_1[0][0]  |
| cropping2d_1 (Cropping2D)       | (None, 65, 320, 3)  | 0      | lambda_1[0][0]        |
| convolution2d_1 (Convolution2D) | (None, 31, 158, 24) | 1824   | cropping2d_1[0][0]    |
| convolution2d_2 (Convolution2D) | (None, 14, 77, 36)  | 21636  | convolution2d_1[0][0] |
| convolution2d_3 (Convolution2D) | (None, 5, 37, 48)   | 43248  | convolution2d_2[0][0] | 
| convolution2d_4 (Convolution2D) | (None, 3, 35, 64)   | 27712  | convolution2d_3[0][0] | 
| convolution2d_5 (Convolution2D) | (None, 1, 33, 64)   | 36928  | convolution2d_4[0][0] |
| flatten_1 (Flatten)             | (None, 2112)        | 0      | convolution2d_5[0][0] |
| dropout_1 (Dropout)             | (None, 2112)        | 0      | flatten_1[0][0]       |
| dense_1 (Dense)                 | (None, 100)         | 211300 | dropout_1[0][0]       |
| dense_2 (Dense)                 | (None, 50)          | 5050   | dense_1[0][0]         |
| dense_3 (Dense)                 | (None, 10)          | 510    | dense_2[0][0]         |
| dense_4 (Dense)                 | (None, 1)           | 11     | dense_3[0][0]         |  


#### 3. Creation of the Training Set & Training Process

After experimenting with game-style racing-line driving, I found
center-lane driving to be much more reliable for autonomous control.
 
I further augmented the data in a manner I had employed in the previous project,
by flipping, shifting and adjusting brightness.

The weighting I applied to the image depending on center, left or right selection
took some experimentation for the tighter corners and played a factor in success.