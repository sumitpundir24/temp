# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[image1]: ./visualize/labels.png "Visualization"
[image11]: ./visualize/labels2.png "Visualization"
[image12]: ./visualize/histogram.png "Visualization"
[image2]: ./visualize/before_processing.png "Grayscaling"
[image21]: ./visualize/after_processing.png "Grayscaling"
[image3]: ./visualize/softmax.png "Softmax score"
[image4]: ./signs/Do-Not-Enter.jpg "Traffic Sign 1"
[image5]: ./signs/sign_test.jpg "Traffic Sign 2"
[image6]: ./signs/slippery_roads.jpg "Traffic Sign 3"
[image7]: ./signs/stop.jpeg "Traffic Sign 4"
[image8]: ./signs/street-warning-sign-23729482.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
This shows the various traffic signs with their corresponing classes
![alt text][image1]
![alt text][image11]
This shows the distribution of data with each class.
![alt text][image12]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because we want to classify images not on the basis of traffic sign color but on the basis of traffic sign shapes. So converting to grayscale helps the network to learn the shapes better.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]    ![alt text][image21]

As a last step, I normalized the image data because it becomes easy for optimizer to find the solution in normalized image.Also nomalization changes the pixels values of images to a common scale, without distorting differences in the ranges of values or losing information.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout Layer     	| Dropout probability = 0.65 	|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 14x14x6. 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16       									|
| RELU					|												|
| Dropout Layer     	| Dropout probability = 0.65 	|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 5x5x16 				|
| Fully connected		| Output = 400        									|
| Fully connected		| Output = 120        									|
| RELU					|												|
| Dropout Layer     	| Dropout probability = 0.65 	|
| Fully connected		| Output = 84        									|
| RELU					|												|
| Fully connected		| Output = 10        									|
| Softmax				|         									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Lenet model given in the lectures. Initially the model is giving very low accuracy. After converting to grayscale and normalizing the data the validation accuracy increases by a small amount. Adding the dropout after 1st convulation layer, 2nd convulational layer and after 2nd fully connected layer increase my validation accuracy to more than 0.93% because adding dropouts decreases the chances of overfitting in the model. The optimizer used is ADAM optimizer, batch size is 128, number of epoch is 40.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.942
* test set accuracy of 0.927

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture I tried was Lenet which was explained in course lesson.


* What were some problems with the initial architecture?

The Lenet is performing good on MNIST datset as discussed in the course but It's giving very low accuracy on the Traffic Sign classification dataset. Actually the model is getting overfitted.


* How was the architecture adjusted and why was it adjusted? 

I added a 3 dropouts layer after 1st convulation layer, 2nd convulational layer and after 2nd fully connected layer in my model.
Also I converted the dataset images to grayscale and normalized them.

* Which parameters were tuned? How were they adjusted and why?

I tried different learning rates. Intially I choosed learning rate to between range (0.001, 0.005). Finally I choosed 0.4 as I think it performed better than others. Also I tried different dropout probability because It helps to avoid overfitting in the model.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
In today's world as the demand of computer vision applications increases normal Image processing processing are giving state of the start results. So, more better techniques like Deep learning is required. CNN performs better when we need to deal with image classification, Object detection, segmentation etc. As this is the problem of image classification so, I think the CNN works well for this problem.  Since our model is getting overfitted while training So adding dropout reduces the complexity of model by randomly dropping the weights of some of the nuerons.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery Road      		| Slippery Road   									| 
| No entry     			| No entry 										|
| Right of the way at the next intersection					| Right of the way at the next intersection											|
| Pedestrians	      		| General caution					 				|
| Stop			| Stop      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.927.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the ln [69] cell of the Ipython notebook.

The below image shows the top 5 scores of each image with their labels.

![alt text][image8] 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


