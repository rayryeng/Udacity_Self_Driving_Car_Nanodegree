# Advanced Lane Finding Project

## Introduction

In this project, the goal is to perform more advanced computer vision algorithms to properly find lane markings on front-facing cameras mounted on cars.  The algorithm will be applied to a test image to ensure it works, then finally applied over a video sequence.

Specifically, the goals / steps of this project are the following given a front-facing image of the road:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
* After, for a stream of images coming of a front-facing camera, we do the following:
    1. Apply a distortion correction to raw images.
    2. Use color transforms, gradients, etc., to create a thresholded binary image.
    3. Apply a perspective transform to rectify binary image ("birds-eye view").
    4. Detect lane pixels and fit to find the lane boundary.
    5. Determine the curvature of the lane and vehicle position with respect to center.
    6. Warp the detected lane boundaries back onto the original image.
    7. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

We will cover each step of the pipeline in more detail throughout this report.

## Code Overview

I decided to take an object-oriented approach and wrap the pipeline into classes.  Specifically, the camera calibration process is separate from the lane detection process so I put this into a class called `CameraCalibration`.  In addition, any utilities to help facilitate the lane detection, such as the colour and gradient detection, the bird's eye view (BEV) image creation and the searching in the BEV image to find the polynomial coefficients for characterising a lane can be found in the `UdacityUtils` class.  Every method is a static method.  Finally, the main pipeline can be found in the `LaneDetection` class which internally does camera calibration and is then ready to accept images of a front-facing camera to determine where the lanes are in an image.

This code can be found in `lane_lines_pipeline` in the `camera calibration.py`, `udacity_utils.py` and `lane_detection.py` files respectively.

Also note that the beginning of our pipeline is on line 174 of `lane_lines_pipeline/lane_detection.py` with the method to the `LaneDetection` class called `apply_pipeline`.  We will begin our tour of the lane detection pipeline here.

## 1. Camera Calibration

The process for was to examine the calibration images of a checkerboard target provided to us and to run through each image to determine where the corner points are.  Each image was taken at different perspectives in order to generalise to a solution.  The goal is to find the intrinsic matrix and the distortion coefficients of the camera and lens so that we can undistort images coming from a live stream of the road when we are finally ready to run this on our road images and video.  This process must be completed first prior to running the lane detection algorithm.  

We know *a priori* that there are a certain number of corner points horizontally and vertically when we take a picture of the checkerboard calibration target.  If we denoted a corner point such that it intersects with two white squares and two black squares, we would have 9 corners horizontally and 6 corners vertically.  Denoting the top left corner using this logic to be the origin `(x, y) = (0, 0)`, we can create a coordinate system where each corner point encountered as we move away from the origin increases by 1.  Therefore, the top right corner would be `(x, y) = (8, 0)` the bottom left corner would be `(x, y) = (5, 0)` and the bottom right corner would be `(x, y) = (8, 5)`.  Because we want to calculate the intrinsics of the camera, we actually need 3D points.  Thankfully, all of the points are parallel to the xy plane, so all `z` points are simply 0.  We now have the coordinate system in 3D.  In 2D, we can use OpenCV's `cv2.findChessboardCorners` to help determine the pixel coordinates of the corresponding corner points so we have a 3D to 2D correspondence.  This is finally used for determining the camera intrinsics and the distortion coefficients of the lens through `cv2.calibrateCamera`.  Therefore, we cycle through each checkerboard image to see if we can locate 9 corner points horizontally and 6 corner points vertically.  If we do, these get added to their own master list of points.  These collections of points then get submitted to `cv2.calibrateCamera` where they are jointly used to finally estimate the intrinsics and distortion coefficients.

Using the intrinsics and distortion coefficients, we can properly undistort images coming off of the camera through `cv2.undistort`.  As an example of this, the image below shows an image of the checkerboard pattern before and after correction.

| Distorted | Corrected | 
|:-:|:--:| 
| ![][orig1]  | ![][image1_zoom] |

## 2. Pipeline (single images)

### Step #1 - Undistort Images
Now that we have properly calibrated our camera, the goal is to perform lane detection and tracking on undistorted images.  As an example, the image below shows one frame that is distorted and the corrected versions.

| Distorted | Corrected | 
|:-:|:--:| 
| ![][example1]  | ![][image2_zoom] |

We can see that the outer edges of the image have either been straightened out or have been removed from view so that straight lines are indeed straight.  Note that we first Gaussian blur the incoming image, found at line 194 in the `lane_lines_pipeline/lane_detection.py` file followed by undistorting the camera image found at line 202.  

### Step #2 - Transform the image into binary using a combination of colour and gradient information

After applying distortion correction to the images, the goal is to perform segmentation using colour and gradient information so that the end result would consist of lanes as well as other objects that match the same colour distribution and shape of the lanes.  However, we can additionally mask out information that isn't pertinent to us for estimating the lanes by simply masking out a polygonal region that coincides with the lanes visible by the vehicle (i.e. the drivable area).

After many trials and tribulations, I used the following combinations of detectors and colour spaces to get the response I needed:

1.  Using a Sobel gradient detector to detect vertical edges (i.e. the `x` direction) where the response is scaled to an image between `[0-255]` then any values between 50 and 255 inclusive become binary true and everything else is binary false.  The kernel size for this operator is 3 x 3.
2.  Using a Sobel gradient detector, we calculate the magnitude of the gradients (i.e. the square root of the vertical edge response squared and horizontal edge response squared), then scale the response image like we did in (1) and also use 50 and 255 as the lower and upper thresholds.  The kernel size for this operator is 3 x 3.
3.  Also using a Sobel gradient detector, we use the vertical and horizontal edge responses to calculate the angle that each edge pixel makes.  Recall that the angle calculated for each edge pixel is the angle *perpendicular* to it.  Also to ensure we get a good angle response, we increase the kernel to a 15 x 15 size.  In order to make the angles non-ambiguous, we take the absolute values of the responses so that we angles from 0 to 90 degrees.  Therefore, by experimentation it was found that edge pixels with an orientation between 40 degrees and 75 degrees are in the range for finding left and right lane pixels.  The binary mask is created like in (1) using the lower and upper thresholds.
4.  Finally, in order to make the pipeline more robust to shadows, colour changes and illumination varying, we transform the image from RGB to the HLS colour space and look at the saturation component of the image.  We noticed that the saturation component is unaffected when the lane is yellow or white.  From experimentation, any saturation values that are between 170 and 255 inclusive are of interest, so we create a binary mask that satisfies this property like we did in (1) using the lower and upper thresholds.  The

Finally, to get the segmentation that I wanted, a pixel should either be a strong vertical like edge in (1), have the right magnitude *and* angle response with (2) and (3) or have the saturation component we want from (4).  If any of the three are satisfied, this is a pixel we want to have a look at.

The figure below shows the undistorted image in the example above and the segmented version.

| Before | After | 
|:-:|:--:| 
| ![][image2_zoom]  | ![][edges1] |

To locate this logic in the code, please navigate to `lane_lines_pipeline/udacity_utils.py` and go to line 166 which is the start of the `combine_colour_and_gradient` function.  This function is used in the lane lines detection pipeline found in `lane_lines_pipeline/lane_detection.py` at line 207.

### 3. Applying a perspective transform to obtain a BEV image of the lanes

In order to concentrate on just the lane lines and ignore the rest of the environment, we should define a polygonal region of interest that only includes the lane lines we want to look at and map this polygonal area to a rectangular region, simulating that we are looking at the lanes up above (i.e. a bird's eye view).  Line 210 does this for us in the aforementioned file.  However, it should be noted that before running the pipeline, we need to know where exactly the polygonal region of interest is prior to obtaining the BEV image.  After much trial and error, shown below are the following pixel locations of the four corners of the polygon from the original image and where they should map to in the final BEV image.
| Source        | Destination   | Location     |
|:-------------:|:-------------:| :------:     | 
| -100, 720     | 100, 720      | Bottom Left  |
| 564, 450      | 100, 0        | Top Left     |
| 716, 450      | 1180, 0       | Top Right    |
| 1380, 720     | 1180, 720     | Bottom Right |

For the source pixels, I made extra exaggeration to capture any ego lanes that could potentially be in the drivable area within the field of view of the car, so I specified the bottom left and bottom right horizontal coordinates to go outside of the image.  The top left and top right points of the polygon were simply chosen by eye.  For destination region where this polygon area will map to, I made sure that the viewable area had a margin of 100 pixels so that the nonlinear mapping from source to target will be mostly limited to this area in order to see mostly everything.  This also allows us to estimate the lanes easier as they won't be so close to the vertical borders of the image.

Using the above source and destination locations, I verified that my perspective transform was working as expected.  I chose an image where the lines aren't parallel but they're along a curve but the point is so that we are easily able to optimise for a set of coefficients describing a parabolic curve and it is indeed so.

![][image3]

### Step #4 - Detect lane pixels in the BEV and fitting a polynomial through them to parameterise a lane

The image below shows the BEV version of the segmented image from Step #2. 

| Before | After | 
|:-:|:--:| 
| ![][edges1]  | ![][image4] |


The process to do this is in two-stages.  For the first frame, we perform a brute-force approach.  This begins at line 217 in the aforementioned file.  The brute-force approach takes the segmented BEV image seen above and examines the bottom half of the image for the moment as it is most likely the case where the lanes will appear to be parallel.  We then compute column-wise sums of the bottom half of this image to generate a one-dimensional signal.  The signal profile that is generated would be bimodal.  As such, we begin our search for fitting a polynomial through the lane lines by first determining where along the bottom of the image to start tracing out the polynomial.  We break up the bottom of the image into two halves as we expect to see one peak in each half.  Wherever the `x` location is for the peak in either of the halves is where we start.  Next for either half, we surround a window and count up how many pixels there are in this window.  If the amount exceeds a threshold, we calculate the average `x` and `y` coordinates within this window for the pixels that are nonzero so that we can start our search here for the next iteration.  This is very similar to the popular [mean-shift algorithm](https://en.wikipedia.org/wiki/Mean_shift) and it can be proven that the average coordinates within a window and constantly recentreing the window is the most optimal solution.  Before we calculate the average `x` and `y` coordinates within the window to decide where to move, we collect every `(x, y)` pair that is nonzero within the previous window.  We repeat the counting of nonzero pixel values, recalculating the centre of the window and collecting all nonzero locations for a maximum number of windows.

This results in a series of `x` and `y` coordinates that best characterise what the lane is.  To close the gap, we fit a second order polynomial through these points that finally charactertise the lane.  Take note that we fit the polynomial such that the `y` or vertical dimension is the independent variable while the `x` or horizontal dimension is the dependent variable.  The reason why is because we expect the lanes to be parallel so there will be more than one possible `x` value mapping to a `y` value should the situation be reversed.

The above approach demonstrates the brute-force approach.  Because video sequences are very similar to each other between frames, we don't need to use brute-force for each frame as it can be computationally intensive.  Instead if we have previous coefficients for the left and right lanes found from the previous frame, we can use these as a starting point when searching for the nonzero lane pixels to calculate the new polynomial coefficients.  Specifically, for each point along the fitted curve using the polynomial coefficients, we can surround a margin of pixels around the curve and find the nonzero pixels that are within all margins for all pixels using the previous curves.  We use these nonzero pixels to recalculate the polynomial coefficients for the lane.  This is done on line 229 of the aforementioned file.

We can switch back to brute-force if we see that we no longer have a sufficient number of nonzero pixels in the current frame that are along the same trajectory as the curve from the previous frame.

To make things even more robust, we have a history of lane curve coefficients for the left and right lanes over the last `n` frames.  This way we can mitigate any noisy curves that get introduced and lean on the cleaner history of the previous frames to smoothen the detected lanes.  Specifically, we simply average each polynomial coefficient over the previous `n` frames so that the trajectory is smoothed over time.

The figure below shows the brute-force logic being applied to localise where the lanes are.  The green windows show the "mean-shift" logic working to isolate out the nonzero pixels that belong to the lane, followed by a yellow curve being plotted in the centroid of the candidate lane lines that tell us where the lanes actually are.

![alt text][image5]

### Step #5 - Calculating the curvature and distance away from the centre of the lane

After we calculate the polynomial coefficients to find the left and right lanes, we can use these to first determine where approximately the vehicle is residing by finding which `(x, y)` coordinates for the left and right lanes belong to the bottom most part of the image.  You can simply specify the height of the image as the `y` coordinate, then find where the horizontal coordinate would appear.  By finding the difference of these pixels, this will roughly give us the width of the lane.  If we assume that the camera is mounted in the centre of the car we can assume that the middle of this image is where the centre of the car would be in pixel coordinates.  We can thus find the difference between the middle of the image and the centroid of the width span of the lane to tell us how far we are off in pixels from the centre of the lane.  Of course this is not meaningful for us so we need actual physical units.  US regulations dictate that the lane width is 3.7 m.  Therefore, we can calculate a scaling factor which gives us a conversion factor from pixels to physical metres.  Simply put, this conversion factor is `3.7 / (lane_right_start_x - lane_left_start_x)` with `lane_left_start_x` and `lane_right_start_x` being the column coordinates where the lane would start given that it's at the bottom of the image.  Once we find the difference between the centre of the lane in pixel coordinates and with the centre of the image, we can multiply by this factor to tell us the final distance in metres.  Take note that you can be on either the left of the centre or the right of the centre.  By taking the difference between lane centre and the car centre.  If the difference is negative, this means we are veering to the left of the centre whereas if the difference is positive, we're veering towards the right.  This distinction is important as we will need to report this for later.  This is done on line 284 from the aforementioned file.

To get the curvature, we convert the pixel coordinates that define each left and right lane (you can simply use the polynomial equation and use each row location from top to bottom and get the corresponding column location) to real world coordinates by using an appropriate conversion factor from pixel to metres.  Once we calculate this new polynomial in physical space, we can use equations of curvature and Calculus that are defined for second-order polynomials to finally give us the curvature we need.  We should choose the point which is located right at the car, so the bottom of the image.  We also need to make sure we convert this bottom coordinate to metres.  Take note tha there are two possible curvatures we can compute.  To mitigate this, we calculate both curvatures and just take the average so approximate the curvature with respect to the centre of the car.  This is done on line 279 from the aforementioned file.

### Step #6 - Showing the lanes and drivable area superimposed on the original image

Finally, to convince ourselves that we did the right thing, we can simply reverse the warping we did to go from BEV to the original image.  Specifically, we can draw the lane lines in the BEV image using the polynomial coefficients detected, we can consider this a closed polygon and fill this area up with some meaningful colour, then take this image and use the reverse mapping from BEV to the original image to give us a mask of the drivable area for the ego vehicle.  We can overlay this on top of the original frame to give us an indication if the lane lines were indeed correct.  We can also add in the curvature and distance from the centre on this image, which we do write in.  This is all done on line 287 of the aforementioned file.  We also provide a sample image showing all of this in the figure below.  Take note that we also distinguish which side away from the centre of the lane we're in.

![][image6]
---

## Pipeline (video)

To tie things off, we demonstrate this on the included project video.  Please consult the link below.

![Link to my video result](test_videos_output/project_video_output.mp4)

You can also find this video in `test_videos_output/project_video_output.mp4`.

---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the hardest things to get right was the segmentation of the colour and gradient information to produce details I needed to isolate out the lane.  It took several hours to figure out the right combination of colour and gradient information to get something that looks remotely close to what I have presented above.  I also had to fiddle around with the hyperparameters a lot to get it to the state that it is in.  Another issue that was a bit hard to deal with was getting the polygonal region correct to isolate out the lanes.  It took several tries with the source and destination coordinates to finally get something that looks decent.  Also the current way of smoothing the polynomial coefficients for each of the lanes can be problematic in the presence of noise.  The simple averaging would not be robust to extreme changes.

With how the polynomial coefficients are calculated, extreme changes like a sharp left or right turn would not fit the lanes properly which is why the challenge video provided would not work.  In order for this to be robust, perhaps a deep learning based solution where it can learn its own set of filters to isolate out what a lane is what isn't would be beneficial.  This would work only if we have a sufficient number of labelled examples which of course are time-consuming to obtain.  We could also take a look at piecewise modelling instead of assuming a second-order polynomial.  Failing that, we could cycle through many orders from 2 up to some high number and see which curve has the most agreement with the lane pixels.  Finally, with the simple averaging of coefficients, a better way to mitigate this would be to gather the last `n` `(x, y)` coordinates detected that are lane pixels, find the average of these and fit a polynomial for these.  This would make things more robust as we are finding the least squares solution that fits the points, and not interfering with the actual coefficients themselves.

[//]: # (Image References)

[image1]: ./images_for_report/calibration_demo.png "Calibration Demo"
[image1_zoom]: ./images_for_report/corrected_zoom_checkerboard.png "Calibration Demo Zoom"
[image2]: ./images_for_report/calibration_demo_road.png "Road Transformed"
[image2_zoom]: ./images_for_report/undistort.png "Calibration Demo"
[image3]: ./images_for_report/test2_bev.png "Bird's Eye View"
[image4]: ./images_for_report/bev_edges.png "Warp BEV"
[image5]: ./images_for_report/lane_image.png "BEV Lanes"
[image6]: ./images_for_report/final_image.png "Output"
[video1]: ./test_videos_output/project_video_output.mp4 "Video"
[orig1]: ./camera_cal/calibration1.jpg "Calibration Image"
[example1]: ./test_images/test2.jpg "Example Input"
[edges1]: ./images_for_report/edges.png "Segmentation"