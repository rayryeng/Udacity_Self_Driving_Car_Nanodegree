# **Finding Lane Lines on the Road** 

## Reflection Report

---

## Introduction

The objective of this first assignment is to get a feeling for how lane lines are detected for the ego vehicle in order to autonomously ensure it stays within its lane for safe navigation.  Specifically, given some sample images from the I-280 freeway from California, the main objective is to use preliminary computer vision algorithms such as the Canny Edge Detector, the Hough Transform and Gaussian blurring to design a processing pipeline to automatically find the left and right ego lanes for the vehicle in these images.  Additionally, given a video stream of the same freeway (which is simply a sequence of images), the goal is to apply the aforementioned pipeline to this video stream.

The main algorithm, as well as a discussion on the results, shortcomings and possible future work are included with this writeup.

---

## Pipeline Design

Using the material learned in the first few lessons, the pipeline is relatively straight forward.  The figure below shows several intermediate images that demonstrate the pipeline from start to finish.  Assume we are given a colour image of the road (top left).

![Pipeline Overview][pipeline]

1. Convert the image into grayscale (top right).  We also blur the image with a small Gaussian kernel to mitigate noise (second row, left).
2. Perform Canny Edge Detection to find strong lines in the grayscale image.  It is assumed that the lane lines would be considered as strong edges (second row, right).
3. Define a polygonal region in front of the ego vehicle that concerns traffic and motion to help isolate out the lanes.  We use this region to mask out any edges from (2) and hope that what is remaining are edges that belong to the left and right lanes (third row, left).
4.  Perform the Hough Transform on the edge region defined in (3) to find strong lines.  After this operation, we hope to see lines that belong to the left and right parts of the ego vehicle lane.  As an additional debug image, the third row right image overlays the lines detected by the Hough Transform that are of interest in the polygon region of interest on the raw Canny Edge Detection performed earlier.
5.  We finally close the gap by using the detected lines from (4).  We must combine disconnected lines found on both sides of the lane to single consolidated lanes and extrapolate the lanes so that the start from the bottom of the image up until the highest point of the polygon region of interest defined in (3).  The bottom left image shows the extrapolated and post-processed lane lines on the Canny Edge Detection image while the bottom right image are the lines superimposed on the original image from the top left.

In the provided notebook, we perform this for both sample images as well as two video streams.  Please consult the notebook for those results.  Naturally, the most interesting part of this project is Step #5, which will will go over next.

## 1. `draw_lines` Modication

To achieve Step #5 in the above pipeline, we modify the `draw_lines` function provided in this project such that for each line from the Canny Edge detector, we calculate the slope of the line.  The slope is useful so that we can ultimately combine disconnected lines together.  Even though the lines aren't connected, they should theoretically have the same slope.  Therefore, using the slope can help us determine which lines belong together on the left side of the lane and right side.  Using the sign of the slope, we can determine which side of the lane we are looking at.  If the slope is negative, we determine this is for the left lane.  If the slope is positive, we determine this is for the right lane.  For each line, we calculate the slope and check the sign.  We thus create two groups of slopes - those that are negative and those that are positive.  It is also imperative that we calculate the corresponding intercept terms for each slope we calculate, ultimately describing the lines in mathematical form.  In other words, `y = mx + b` with `m` and `b` being the slope and intercept respectively.  Therefore, we will have two groups of slopes and corresponding intercept terms - one for the left and right.  Naturally due to noise, there should be a way to aggregate all of the slopes and intercepts together for both sides of the lane independently and reduce the influence of outliers.  To facilitate this, a simple median was used.  Therefore, for each side of the lane, we calculate the median slope and intercept which finally give us the equation of the line that is representative of those lane lines.  The final goal is to draw one consolidated lane from the bottom of the image up to the top most part of the polygon region of interest.  This is done by simply using the row coordinate for the bottom of the image and top most part of the polygon, and with the corresponding slope and intercept we solve for the column coordinates required to satisfy the equation of the line.  

## 2. Potential Shortcomings for the Pipeline

Though the pipeline does achieve decent results, it is not without its problems:

1. There are a significant number of hyperparameters to tune in order for this to work with the images and video streams for this assignment.  In particular, the kernel size for the Gaussian blur, the low and high thresholds for the Canny Edge Detector, the hyperparameters for the Hough Transform and the polygonal region of interest to look for the lane edges.  The final parameters seen in the submitted notebook are specifically tuned for the data provided.  It is inevitable that other highway footage will cause the pipeline to fail.
2. The pipeline assumes that the lane edges are straight.  Should the ego vehicle experience any slight turning where the lane edges are curved, this pipeline will ultimately fail to trace over the lanes correctly
3. Though we were lucky with the data provided to us, the lanes can appear in different colours, and converting to grayscale will lose the powerful semantic information that is provided by colour cues.  As an example, yellow lanes in grayscale appear to blend with the road so we miss out on those lines.

## 3. Suggestions to improve the pipeline

1. Because of the reliance on hyperparameters, if there is enough data available, perhaps a deep learning based approach would be better as the framework is designed to adaptively learn the best set of feature representations to localise where the lanes are regardless of scene composition.  To make this truly robust, we would perhaps employ semantic segmentation to the fold, but the accuracy of the labelled data is paramount.
2. If we can modify the pipeline to handle curved lanes while turning, this would make the pipeline more robust.
3.  If we can use colour information to allow the lanes to appear in another representation that would be immune to shadows, weather changes and colour changes, this will greately help make the pipeline more robust.

[//]: # (Image References)

[pipeline]: ./images_for_report/debug_plot_solidYellowCurve.jpg "Pipeline Overview"