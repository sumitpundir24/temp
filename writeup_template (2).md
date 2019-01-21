# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_architecture.png "Model Visualization"
[image2]: ./examples/center.png "center"
[image3]: ./examples/left.png "left"
[image4]: ./examples/right.png "right"
[image5]: ./examples/cropped.png "cropped"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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
* run1.mp4 The video on track1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with and 5x5 and 3x3 filters sizes and depths between 24 and 64. The model is using the pooling layer of pool size 2x2.

The model includes RELU activation in intermediate layers to introduce nonlinearity, and the data is normalized. I used the nvidia model because it's simple and also performs well on self-driving car training. I chnaged the shape of input to (75,155).

![alt text][image1]


#### 2. Attempts to reduce overfitting in the model

I tried several dropout layers in order to reduce overfitting. Also batch normalization also reduce overfitting to small extent. When I tested the model, the car go out of the 
track in the lake. After removing batch normalization and dropout layers the model performed well without the car being going of the track.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Also the dropout with different probabilties is tried. Used 'MSE' loss function. Used batch size of 32.

#### 4. Appropriate training data

I used the training data provided by the udacity. It contains images from multiple angles. driving_log.csv contains the paths to images and steering angles.  It contains combination of center lane, left and right side recovery from the road inages. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to see the bias-variance tradeoff.

My first step was to use a convolution neural network model inspired by NVIDIA End to End Learning for Self-Driving Cars paper. I thought this model might be appropriate as it provided and tested by nvidia.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

I found that my first model had a low loss on the training set but a high loss on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout and batchnormalisation layer.

Then I ran the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. to improve the driving behavior in these cases, I augment the data by flipping the image 
and correct angle distortion by 0.2. Still the car was going out of the track.

At the end of the process, I simply removed all the dropout and batch normalization layers then, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes.

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(75,155,3)))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


#### 3. Creation of the Training Set & Training Process

I used the sample dataset provided by udacity. The images are normalized before feeding them into network.

To augment the data sat, I also flipped images and angles.

Training images contains 4 types of images :
1. center
2. left
3. right
4. center flipped

Correction factor of 0.2 is applied on left and right images. Correction factor of 0.2 is added to the left image. Correction factor of 0.2 is subtracted from the right image.


Center

![alt text][image2]

Left

![alt text][image3]

Right

![alt text][image4]

Cropped

![alt text][image5]



I finally randomly shuffled the data set and put 22% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2. I used an adam optimizer so that manually training the learning rate wasn't necessary.
