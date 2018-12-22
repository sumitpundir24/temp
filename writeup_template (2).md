## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./readme_images/undistort_s.png "Undistorted"
[image2]: ./readme_images/undistort_on_test.png "Road Transformed"
[image3]: ./readme_images/color_x_graident.png "Binary Example"
[image4]: ./readme_images/perspective_transform.png "Warp Example"
[image5]: ./readme_images/identify_lane_pixels.png "Fit Visual"
[image6]: ./readme_images/final_plot.png "Output"
[image7]: ./readme_images/src_des_points.png "Output"
[image8]: ./readme_images/dynamic_search.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function cam_callibrate() for camera callibaryion and cal_undistort() for image distortion correction.
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I applied the distortion correction to one of the test images like this one. I compute the camera matrix and distortion co-efficients to undistort the image.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The functon pipeline() is used to give the thresholded binary image.
COLOR TRANSFORMATION :
It creates a thresholded binary image. The R channel of RGB, S and L channle of HLS is used for color transformation. R channel thresholds are used as they detect yellow lines well, S channle thresholds are used to differentiate between yellow and white lines and L channel is good for avoiding shadows as studied from the lecture.

GRAIDENT TRANSORMATION :
The function abs_sobel_thresh () to find the sobel-x along X-axis is applied on the binary thresholded image.
It is used along X-axis since lane lines are mostly vertical.


The COLOR transformation and gradient transformation are combined together to form a final thresholded image.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Straight lines images are used for perspective transform. Four source points and four destination points are selected. Destination points are selected in a way to get the bird view of a road. 
![alt text][image7]

perform_perspective_transform() function is used to calculate the transformation matrix. Inverse transformation matrix is also calculated which helps in getting the map to original image.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The fucntion find_lane_pixels() identify the lane pixels and fit there positions in polynomials.
The steps used to identify lane are :
Step 1 : Histogram calculation along the X axis. function calculate_histogram() calculates the histogram.
Step 2 : Find the peaks of the left and right side of the lane in histogram. And divide the image into two windows (left and right).
Step 3 : Count all non-zero points present in the window.
Step 4 : Fit the polynomial using np.polyfit().

The polynomial fit done again on the same points to transform pixels into meters for calculation of curvature.

![alt text][image5]

Since consecutive frames is likely to have lane lines in roughly similar manner so search around prevously detected lane line.
Searching around previosly detected lane line Since consecutive frames are likely to have lane lines in roughly similar positions, we search around a margin of 100 pixels of the previously detected lane lines using the function search_around_poly().

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Two things are calculated :
1. Curvature of the road : Function measure_curvature_real() calculates the radius of curvature. Defined conversions in x and y from pixels space to meters. Polynomial fit is calculated in that space. Average of two lines is used a radius of curavture.

2. Position of the vehicle : Function car_position_road() calculates the position of car. Defined conversions in x and y from pixels space to meters. Calculated the position of line beginening from the bottom of the image. Comparing it with middle of image helps in finding the car position assuming car position is in the middle of the car.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally Inverse transform is applied and output is presented on the final image.
The process image is combined with original image.
The function show_final_image_with_lanes() implements this.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)
The function pipeline_final() is the complete pipeline of all the steps explained above.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
 
Challanges and Issues :
A lot of experiment needs to be done in gradient and color thresholding. I tested my results on various values of gradient and thresholding. I focused first on frames where color changes and large shadows are present. So, I've take a threshold to handle those frames first.
I also spend more time on choosing the source and destination points in the perspective transform. Choosing these points improved the performance of pipeline a little bit.
I tried my pipeline on challange video but it fails after completeing 51 % frames. I'll try to make my pipeline better to work on that video also.

Improvements :
The pipeline fails at the harder challenge video. So, I think it can be improved by taking a better perpective tranform by choosing a smaller section of area from image to avoid high turns in the video. 

I think usage of deep nueral networks instead of image processing techniques can do much better in detecting lanes.


