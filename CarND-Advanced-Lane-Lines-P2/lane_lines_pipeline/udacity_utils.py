"""
These are a set of tools from the Udacity Self-Driving Car Nanodegree
that were made available and taken from the lectures

These are helper methods that will help complete the Advanced Lane Driving
project
"""
import cv2
import numpy as np


class UdacityUtils(object):
    @staticmethod
    def abs_sobel_thresh(img, reverse_channels=True,
                         orient='x', thresh_min=0, thresh_max=255):
        """Calculates the Sobel edge response given an image.

        The Sobel edge response is scaled to the [0-255] range, then
        thresholded so that any values that are within a certain range are
        considered edges.

        Args:
            img (numpy.ndarray): Input image - can be colour or grayscale
            reverse_channels (bool): Specify whether the colour channels
                                     are in BGR format (True) or RGB
                                     format (False)
            orient (str): A single character determining which direction the
                          gradient should be taken in - One of 'x' or 'y'
            thresh_min (int): Lower range of the gradient to be considered an
                              edge
            thresh_max (int): Upper range of the gradient to be considered an
                              edge

        Returns:
            A response image that determines which pixels are edges
        """
        # 1) Convert to grayscale
        if len(img.shape) == 3:
            flag = cv2.COLOR_BGR2GRAY if reverse_channels else cv2.COLOR_RGB2GRAY
            gray = cv2.cvtColor(img, flag)
        else:
            gray = img.copy()
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sob = cv2.Sobel(gray, -1, int(orient == 'x'), int(orient == 'y'))
        # 3) Take the absolute value of the derivative or gradient
        sob = np.abs(sob)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        sob = ((255 / sob.max()) * sob).astype(np.uint8)
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary_output = np.logical_and(
            sob >= thresh_min, sob <= thresh_max)
        # 6) Return this mask as your binary_output image
        return binary_output

    @staticmethod
    def mag_thresh(img, reverse_channels=True, sobel_kernel=3,
                   mag_thresh=(0, 255)):
        """Calculates the Sobel magnitude edge response given an image.

        The Sobel magnitude edge response is scaled to the [0-255] range, then
        thresholded so that any values that are within a certain range are
        considered edges.

        Args:
            img (numpy.ndarray): Input image - can be colour or grayscale
            reverse_channels (bool): Specify whether the colour channels
                                     are in BGR format (True) or RGB
                                     format (False)
            sobel_kernel (int):  Size of the Sobel kernel
            thresh (tuple of int): Lower and upper range of the gradient
                                   magnitude to be considered an edge

        Returns:
            A response image that determines which pixels are edges
        """
        # 1) Convert to grayscale
        if len(img.shape) == 3:
            flag = cv2.COLOR_BGR2GRAY if reverse_channels else cv2.COLOR_RGB2GRAY
            gray = cv2.cvtColor(img, flag)
        else:
            gray = img.copy()
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude
        mag = np.sqrt(np.square(sobelx) + np.square(sobely))
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        mag = (255 * mag / np.max(mag)).astype(np.uint8)
        # 5) Create a binary mask where mag thresholds are met
        binary_output = np.logical_and(
            mag >= mag_thresh[0], mag <= mag_thresh[1])
        # 6) Return this mask as your binary_output image
        return binary_output

    @staticmethod
    def dir_threshold(img, reverse_channels=True, sobel_kernel=3,
                      thresh=(0, np.pi/2)):
        """Calculates the Sobel angle response given an image.

        The Sobel angle edge response is scaled to the [0-255] range, then
        thresholded so that any values that are within a certain range are
        considered edges.

        Args:
            img (numpy.ndarray): Input image - can be colour or grayscale
            reverse_channels (bool): Specify whether the colour channels
                                     are in BGR format (True) or RGB
                                     format (False)
            sobel_kernel (int):  Size of the Sobel kernel
            thresh (tuple of int): Lower and upper range of the angles
                                   to be considered an edge

        Returns:
            A response image that determines which pixels are edges
        """
        # Apply the following steps to img
        # 1) Convert to grayscale
        if len(img.shape) == 3:
            flag = cv2.COLOR_BGR2GRAY if reverse_channels else cv2.COLOR_RGB2GRAY
            gray = cv2.cvtColor(img, flag)
        else:
            gray = img.copy()
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the x and y gradients
        sobelx = np.abs(sobelx)
        sobely = np.abs(sobely)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        ang = np.arctan2(sobely, sobelx)
        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.logical_and(ang >= thresh[0], ang <= thresh[1])
        # 6) Return this mask as your binary_output image
        return binary_output

    @staticmethod
    def hls_select(img, reverse_channels=True, thresh=(0, 255)):
        """Calculates the saturation channel and thresholds it given a colour
        image.

        We convert a colour image into the HLS space, then threshold it so that
        any values that are within a certain range are considered edges.

        Args:
            img (numpy.ndarray): Input colour image
            reverse_channels (bool): Specify whether the colour channels
                                     are in BGR format (True) or RGB
                                     format (False)
            thresh (tuple of int): Lower and upper range of the saturation
                                   values to be considered an edge

        Returns:
            A response image that determines which pixels are edges
        """
        # 1) Convert to HLS color space
        flag = cv2.COLOR_RGB2HLS if reverse_channels else cv2.COLOR_BGR2HLS
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the S channel
        binary_output = np.logical_and(
            hls[..., 2] > thresh[0], hls[..., 2] <= thresh[1])
        # 3) Return a binary image of threshold result
        return binary_output

    @staticmethod
    def combine_colour_and_gradient(img):
        """Method for taking in a lane image, and finding good edges that
           allow us to find lane edges

        Args:
            img (numpy.ndarray): Input image of a front-facing camera in a car

        Returns:
            A decent edge detection response showing us the lanes and other
            edges
        """

        # Apply each of the thresholding functions
        gradx = UdacityUtils.abs_sobel_thresh(img, orient='x',
                                              thresh_min=50,
                                              thresh_max=255)
        mag_binary = UdacityUtils.mag_thresh(img, sobel_kernel=3,
                                            mag_thresh=(50, 255))
        dir_binary = UdacityUtils.dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
        hls_binary = UdacityUtils.hls_select(img, thresh=(170, 255))
        
        combined = np.zeros_like(dir_binary)
        combined[(gradx == 1 | ((mag_binary == 1) & (dir_binary == 1))) | hls_binary == 1] = 1
        return combined

    @staticmethod
    def fit_polynomial(binary_warped, nwindows=9, margin=100, minpix=50,
                       debug=False):
        """Brute-force search for fitting a polynomial to find the lanes in a
           BEV image

        Args:
            binary_warped (numpy.ndarray): BEV image that may contain lanes
            nwindows (int): Number of sliding windows
            margin (int): Width of the windows +/- this amount
            minpix (int): Minimum number of pixels found to recentre window
            debug (bool): Debug mode to show the lanes detected in the BEV

        Returns:
            A tuple that contains the polynomial coefficients for the left and
            right lane, the actual x points that define the left and right lane,
            the y points that define both lanes and the debug image
        """
        def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50,
                             debug=False):
            """
            Internal Method - Calculates the pixels that belong to a lane
            in the warped BEV image of a front-facing camera image from a car

            Args:
                binary_warped (numpy.ndarray): A warped binary input image
                                               showing us a BEV of the road
                nwindows (int): Number of sliding windows
                margin (int): Width of the windows +/- this amount
                minpix (int): Minimum number of pixels found to recentre window
                debug (bool): Enable to return a debug image (more details below)

            Returns:
                A tuple representing the x and y locations for the left lane,
                the x and y locations for the right lane and a debug image
                telling us how we've tracked the lane locations
            """
            # Take a histogram of the bottom half of the image
            histogram = np.sum(
                binary_warped[binary_warped.shape[0]//2:, :], axis=0)
            # Create an output image to draw on and visualize the result
            out_img = None
            if debug:
                out_img = np.dstack((binary_warped, binary_warped, binary_warped))
                out_img = out_img.astype(np.uint8)
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]//2)
            leftx_base = np.argmax(histogram[100:midpoint]) + 100
            rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint

            # Set height of windows - based on nwindows above and image shape
            window_height = np.int(binary_warped.shape[0]//nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated later for each window in nwindows
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                ### Find the four below boundaries of the window ###
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                # Draw the windows on the visualization image
                if debug:
                    cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                                  (win_xleft_high, win_y_high), (0, 255, 0), 2)
                    cv2.rectangle(out_img, (win_xright_low, win_y_low),
                                  (win_xright_high, win_y_high), (0, 255, 0), 2)

                ### Identify the nonzero pixels in x and y within the window ###
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                    nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                    nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                ### If you found > minpix pixels, recenter next window ###
                ### (`rightx_current` or `leftx_current`) on their mean position ###
                if len(good_left_inds) > minpix:
                    leftx_current = int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices (previously was a list of lists of pixels)
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            return leftx, lefty, rightx, righty, out_img

        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = \
            find_lane_pixels(binary_warped, nwindows, margin, minpix, debug)

        ### Fit a second order polynomial to each using `np.polyfit` ###
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(
            0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        ## Visualization ##
        # Colors in the left and right lane regions
        if debug:
            out_img[lefty.astype(np.int), leftx.astype(np.int)] = [255, 0, 0]
            out_img[righty.astype(np.int), rightx.astype(np.int)] = [0, 0, 255]

            # Also draw the lanes on the image too
            out_img[ploty.astype(np.int), left_fitx.astype(np.int)] = [0, 255, 255]
            out_img[ploty.astype(np.int), right_fitx.astype(np.int)] = [0, 255, 255]

        return left_fit, right_fit, left_fitx, right_fitx, ploty, out_img

    @staticmethod
    def search_around_poly(binary_warped, left_fit, right_fit, margin=100, debug=False):
        """Look-ahead search for fitting a polynomial to find the lanes in a
            BEV image.  Use this so that we don't have to use brute-force unless
            we really need to

        Args:
            binary_warped (numpy.ndarray): BEV image that may contain lanes
            left_fit (numpy.ndarray): The polynomial coeffs for the left lane
            right_fit (numpy.ndarray): The polynomial coeffs for the right lane
            margin (int): Width of the margin around the previous polynomial
                            to search
            debug (bool): Debug mode to show the lanes detected in the BEV

        Returns:
            A tuple that contains the polynomial coefficients for the left and
            right lane, the actual x points that define the left and right lane,
            the y points that define both lanes and the debug image
        """

        def fit_poly(img_shape, leftx, lefty, rightx, righty):
            """ Internal method: Calculate the polynomials and the points
                along them for the left and right lanes
            """
            ### Fit a second order polynomial to each with np.polyfit() ###
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            # Generate x and y values for plotting
            ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
            ### Calc both polynomials using ploty, left_fit and right_fit ###
            left_fitx = np.polyval(left_fit, ploty)
            right_fitx = np.polyval(right_fit, ploty)

            return left_fit, right_fit, left_fitx, right_fitx, ploty

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###

        # Obtain lane curvature from previous frame and find all of
        # the lane curve points for each lane
        left_fitx = np.polyval(left_fit, nonzeroy)
        right_fitx = np.polyval(right_fit, nonzeroy)

        # Find all points that are within margin tolerance horizontally for each
        # lane pixel as we traverse from bottom to top
        left_lane_inds = ((nonzerox >= left_fitx - margin) &
                          (nonzerox < left_fitx + margin)).nonzero()[0]
        right_lane_inds = ((nonzerox >= right_fitx - margin) &
                           (nonzerox < right_fitx + margin)).nonzero()[0]

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # If there are not enough points to reliably make an estimate,
        # quit
        if len(lefty) < 10 or len(righty) < 10:
            return None

        # Fit new polynomials
        left_fit_new, right_fit_new, left_fitx, right_fitx, ploty = fit_poly(
            binary_warped.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = None
        if debug:
            out_img = np.dstack(
                (binary_warped, binary_warped, binary_warped))*255
            out_img = out_img.astype(np.uint8)
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds],
                    nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds],
                    nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array(
                [np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array(
                [np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array(
                [np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array(
                [np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            # Plot the polynomial lines onto the image
            out_img[ploty.astype(np.int), left_fitx.astype(np.int)] = [0, 255, 255]
            out_img[ploty.astype(np.int), right_fitx.astype(np.int)] = [0, 255, 255]
            ## End visualization steps ##

        return left_fit_new, right_fit_new, left_fitx, right_fitx, ploty, out_img

    @staticmethod
    def warp_image(img, vertices, reverse=False):
        """Given an image and a polygonal region of interest, warp the image so
           that the region of interest becomes a BEV image spanning the original
           image dimensions.  Similarly, given a BEV image, you can reverse the
           effects.

        Args:
            img (numpy.ndarray): Input image
            vertices (list of list): A list of four elements with each element
                                     being two elements which are x, y
                                     coordinates.  These define a polygon to
                                     isolate out the region where our lanes are
            reverse (bool): Instead of going to BEV, we can reverse the BEV.
                            Set this to True to do this.

        Returns:
            A warped image such that the corners of the polygonal region of
            interest map to the four corners of the original image dimensions
        """
        src = np.float32(vertices)
        
        # The output region is such that it spans the entire image by
        # we remove the first and last 100 columns
        height = img.shape[0]
        width = img.shape[1]
        dst = np.float32([[100, height], [100, 0], [width - 100, 0], [width - 100, height]])
        if not reverse:
            M = cv2.getPerspectiveTransform(src, dst)
        else:
            M = cv2.getPerspectiveTransform(dst, src)

        img_shape = img.shape[:2][::-1]
        warped = cv2.warpPerspective(img, M, img_shape, flags=cv2.INTER_LINEAR)

        return warped

    @staticmethod
    def gaussian_blur(img, kernel_size=3):
        """Applies a Gaussian Noise filter to an image

        Args:
            img (numpy.ndarray): Input image
            kernel_size (int): Size of the kernel

        Returns:
            A smoothed image using the Gaussian kernel
        """
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    @staticmethod
    def draw_final_lane_image(undist, warped, vertices, left_fitx, right_fitx,
                              ploty, curvature, dist_from_centre):
        """ Draw the final lane image that shows the ego vehicle drivable lane
            the curvature and the distance away from the centre

        Args:
            undist (numpy.ndarray): The undistorted road image
            vertices (list of list): A list of four elements with each element
                                     being two elements which are x, y
                                     coordinates.  These define a polygon to
                                     isolate out the region where our lanes are            
            warped (numpy.ndarray): BEV image of the road
            left_fitx (numpy.ndarray): The x coordinates defining the left lane
            right_fitx (numpy.ndarray): The x coordinates defining the right lane
            ploty (numpy.ndarray): The y coordinates defining the lanes
            curvature (float): The curvature to write to the screen
            dist_from_centre (float): The distance from the centre of the ego lane

        Returns:
            The final lane image showing the drivable lane area, curvature and
            distance away from the centre
        """

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix
        newwarp = UdacityUtils.warp_image(color_warp, vertices, True)
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        # Write the relevant information to the screen
        cv2.putText(result, 'Radius of Curvature: {:.2f} m'.format(curvature),
                    (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)

        # Determine if we're on the left or right of the lane centre
        position = "left" if dist_from_centre < 0 else "right"
        dist_from_centre = abs(dist_from_centre)
        cv2.putText(result, 'Vehicle is {:.2f} m {} off centre'.format(dist_from_centre, position),
                    (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)
        return result
