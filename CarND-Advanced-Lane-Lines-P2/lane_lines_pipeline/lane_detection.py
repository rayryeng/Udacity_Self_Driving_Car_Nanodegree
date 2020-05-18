import cv2
import numpy as np
import matplotlib.pyplot as plt
from .camera_calibration import CameraCalibration
from .udacity_utils import UdacityUtils


class LaneDetection(object):
    """
    Class definition for detecting a lane given images along the road
    """

    def __init__(self, camera_calibration_dir, vertices, smoothing_window_size=5,
                 nwindows=9, margin=100, minpix=50, debug=False):
        """Construct a Lane Detection object

        Args:
            camera_calibration_dir (str): Path to the images containing the
                                          the checkerboard images for calibration
            vertices (list of list): A list of four elements with each element
                                     being two elements which are x, y
                                     coordinates.  These define a polygon to
                                     isolate out the region where our lanes are
            smoothing_window_size (int): Smoothing window size.  Smooth the
                                         trajectory over the last n frames
            nwindows (int): Number of sliding windows for brute-force search
            margin (int): Width of the windows +/- this amount for lane search
            minpix (int): Minimum number of pixels found to recentre window
                          for brute-force search
            debug (bool): For debug mode
        """
        
        # Did we detect lines?
        self._detected = False

        # Keeps track of the total number of frames we've seen so far
        self._frame_counter = 0

        # polynomial coefficients averaged over the last n iterations
        self._fit_left = None
        self._fit_right = None
        self._best_fit_left = None
        self._best_fit_right = None

        # polynomial coefficients for the most recent fit
        self._current_fit_left = None
        self._current_fit_right = None

        # radius of curvature of the line in metres
        self._radius_of_curvature = None

        # distance in metres of the vehicle centre from the centre
        # of the ego lane
        self._line_base_pos = None
        
        # The x coordinate in the image telling us where the lane
        # starts (from the bottom of the image)
        self._lane_right_start = None
        self._lane_left_start = None        
        
        # Camera calibration directory
        self._camera_calibration_dir = camera_calibration_dir

        # Perform calibration to get parameters
        self._calibration_obj = CameraCalibration(self._camera_calibration_dir)
        self._calibration_obj.perform_calibration()
        
        # Hyperparameters captured from constructor
        self._nwindows = nwindows
        self._margin = margin
        self._minpix = minpix
        self._vertices = vertices
        self._smoothing_window_size = smoothing_window_size
        self._debug = debug

        # Width of the lane is assumed to be 3.7 m
        self._lane_width = 3.7

        # Height of the lane is assumed to be 30 m
        self._lane_height = 30

        # Number of relevant pixels in the x and y direction for calculating
        # curvature
        self._x_relevant = 720
        self._y_relevant = 700

    def reset(self):
        """Resets everything back to the initial state
        """
        self._detected = False

        self._frame_counter = 0

        self._fit_left = None
        self._fit_right = None
        self._best_fit_left = None
        self._best_fit_right = None

        self._current_fit_left = None
        self._current_fit_right = None

        self._radius_of_curvature = None

        self._line_base_pos = None
        
        self._lane_right_start = None
        self._lane_left_start = None  

    def __distance_from_centre(self, width):
        """ Internal method to calculate the distance to the centre of the lane
        This assumes the camera is mounted in the centre of the vehicle and the
        road lane is 3.7m wide

        Args:
            width (int): Width of the image

        Returns:
            The distance from the centre in real world coordinates (metres)
        """

        # Find the scaling factor that converts the distance between the lanes
        # equating this to 3.7m
        normalization = self._lane_width / (
            self._lane_right_start - self._lane_left_start)

        # Calculate the lane centre in pixel coordinates
        lane_center = (self._lane_left_start + self._lane_right_start) / 2.0
        car_center = width / 2.0  # assume camera mounted in the centre

        # Find the difference from the centre in pixel coordinates then
        # convert to metres
        distance = (lane_center - car_center) * normalization
        return distance

    def ___get_curvature(self, ploty, left_fitx, right_fitx):
        """ Internal method to calculate the distance to the centre of the lane
        This assumes the camera is mounted in the centre of the vehicle and the
        road lane is 3.7m wide

        Args:
            ploty (numpy.ndarray): The array of y points that define the lane
                                   span vertically
            left_fitx (numpy.ndarray): The array of x points that define the
                                       left lane span horizontally
            right_fitx (numpy.ndarray): The array of x points that define the
                                        right lane span horizontally

        Returns:
            The curvature of the lane in real world coordinates (metres)
        """
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = self._lane_height / self._y_relevant  # metres / pixel in y
        xm_per_pix = self._lane_width / self._x_relevant  # metres / pixel in x

        # Fit a second order polynomial to pixel positions in each lane line
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(
            ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        # Also remember we must convert the pixel coordinate to metres
        y_eval = np.max(ploty) * ym_per_pix

        # Calculate the new radii of curvature
        Aleft, Bleft = left_fit_cr[:2]
        Aright, Bright = right_fit_cr[:2]
        left_curverad = (1 + (2 * Aleft * y_eval + Bleft)** 2)**1.5 / (np.abs(2 * Aleft))
        right_curverad = (1 + (2 * Aright * y_eval + Bright)** 2)**1.5 / (np.abs(2 * Aright))

        # To consolidate the two together, simply find the average
        return (left_curverad + right_curverad) / 2.0

    def apply_pipeline(self, img):
        """ Apply Lane Detection Pipeline to input image

            Given an input image of the road, the output will be the same image but
            the left and right lanes are marked, the area between the lanes are
            marked in green and the radius of curvature and showing how much we are
            off from the centre of the lane is written in the image

            Args:
                img (numpy.ndarray): Input image

            Returns:
                An copy of the input image with the lanes marked, the drivable
                area shaded and the off-centre distance and curvature shown.
                We also return a dictionary of debug results if debugging is
                enabled.  These both are returned as a tuple.  If debugging is
                not enabled, the second element of this tuple is None.
            """
        
        ### Step #0 - Gaussian Blur the image to reduce noise
        blur = UdacityUtils.gaussian_blur(img, kernel_size=5)

        if self._frame_counter == 0:
            # Create storage the last n polynomial coefficients
            self._fit_left = np.zeros((self._smoothing_window_size, 3))
            self._fit_right = np.zeros((self._smoothing_window_size, 3))

        # Store debug results
        results = {}
        
        ### Step #1 - Apply distortion correction to the image
        undist = self._calibration_obj.undistort_image(blur)
        if self._debug:
            results['undist'] = undist

        ### Step #2 - Get edge image
        edges = UdacityUtils.combine_colour_and_gradient(undist)

        ### Step #3 - Get the BEV of the edges
        edges = edges.astype(np.float32)
        if self._debug:
            results['edges'] = edges        
        bev = UdacityUtils.warp_image(edges, self._vertices)
        if self._debug:
            results['bev_edges'] = bev
            results['bev'] = UdacityUtils.warp_image(undist, self._vertices)

        ### Step #4 - Sanity check
        # If we don't detect a line, use brute-force
        out_img = None
        if not self._detected:
            left_fit, right_fit, left_fitx, right_fitx, ploty, out_img = \
            UdacityUtils.fit_polynomial(bev, self._nwindows, self._margin,
                                        self._minpix, self._debug)
            self._frame_counter += 1
            self._detected = True
            self._recent_xfitted_left = left_fitx
            self._recent_xfitted_right = right_fitx     
        # We are fairly confident that the previous frame and this frame are
        # similar - just use a look-ahead filter
        else:
            ret = UdacityUtils.search_around_poly(bev, self._best_fit_left,
                                                  self._best_fit_right,
                                                  self._margin, self._debug)
            if ret is None:
                # Use the same stuff as we did before
                left_fit = self._fit_left[-1]
                right_fit = self._fit_right[-1]
                left_fitx = self._recent_xfitted_left
                right_fitx = self._recent_xfitted_right
                ploty = np.linspace(0, bev.shape[0] - 1, bev.shape[0])
                self._detected = False
            else:
                left_fit, right_fit, left_fitx, right_fitx, ploty, out_img = ret
                self._detected = True
                self._frame_counter += 1
        
        if self._debug:
            results['lane_image'] = out_img
        
        ### Step #5 - Find the best lane line coefficients for each side
        ### Only do this if we have a good detection
        # Store the most recent lane coefficients for the fitted lanes
        # Stack them up if we haven't reached the window size
        if self._detected:
            if self._frame_counter <= self._smoothing_window_size:
                self._fit_left[self._frame_counter - 1] = left_fit
                self._fit_right[self._frame_counter - 1] = right_fit
            else:
                # If we've reached the window size, remove the earliest entry
                # and insert the latest one
                self._fit_left[:-1] = self._fit_left[1:]
                self._fit_left[-1] = left_fit
                self._fit_right[:-1] = self._fit_right[1:]
                self._fit_right[-1] = right_fit
            
        # Store the left and right lane coefficients for the current frame
        self._current_fit_left = left_fit
        self._current_fit_right = right_fit

        # Find the average curve coefficients over the n previously
        # detected frames
        n = self._frame_counter
        if n >= self._smoothing_window_size:
            n = self._smoothing_window_size

        # Find the average polynomial coefficients
        self._best_fit_left = np.mean(self._fit_left[:n], axis=0)
        self._best_fit_right = np.mean(self._fit_right[:n], axis=0)

        ### Step #6 - Compute radius of curvature
        self._radius_of_curvature = self.___get_curvature(ploty, left_fitx, right_fitx)

        ### Step #7 - Compute distance away from the centre
        self._lane_left_start = np.polyval(self._best_fit_left, img.shape[0])
        self._lane_right_start = np.polyval(self._best_fit_right, img.shape[0])
        self._line_base_pos = self.__distance_from_centre(img.shape[1])

        # Get the final rendered image with our information and return it
        output_image = UdacityUtils.draw_final_lane_image(undist, bev, self._vertices,
                                                left_fitx, right_fitx, ploty,
                                                self._radius_of_curvature,
                                                self._line_base_pos)
        if self._debug:
            return output_image, results
        else:
            return output_image, None