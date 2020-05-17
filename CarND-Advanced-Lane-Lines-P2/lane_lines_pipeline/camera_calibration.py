from glob import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class CameraCalibration(object):
    """Performs camera calibration given a set of checkerboard images

    Given a path to a directory containing calibration images, we will
    use OpenCV's camera calibration tools to give us the intrinsic matrix
    and the distortion coefficients to help undistort images further
    down the line.  Once you run the `perform_calibration` method after
    you create an instance of this class, two public attributes will be
    made available for use in undistorting image

    Attributes:
        K (numpy.ndarrray): Camera intrinsic matrix
        dist (numpy.ndarray): Distortion coefficients
    """

    def __init__(self, path_to_images):
        """Construct a Camera Calibration object

        Args:
            path_to_images (str): Path to the checkerboard images - Extension
                                  should be JPG
        """
        self._path = path_to_images
        self._filenames = list(
            sorted(glob(os.path.join(path_to_images, '*.jpg'))))
        self.K = None
        self.dist = None
        self._num_images_passed = None

    def perform_calibration(self, debug=False):
        """Perform camera calibration on the checkerboard images

        Args:
            debug (bool): Enable debug mode to show checkerboard corner
                          detections
        Returns:
            True if successful and False otherwise
        """
        if len(self._filenames) == 0:
            print('ERROR: Could not find any checkerboard images')
            return False

        if debug:
            n = len(self._filenames)
            m = int(np.sqrt(n))
            if n != m ** 2:
                m += 1
            plt.figure(figsize=(16, 12))

        # Expected number of corners in each direction
        nx, ny = 9, 6
        self._num_images_passed = 0

        # Store the object points in 3D space and corresponding
        # 2D corners found for each image
        objpoints, imgpoints = [], []

        # Generate object points first - they're all going to be the same
        # for each image
        objp = np.zeros((nx * ny, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # For each file...
        img_shape = None
        count = 1
        for i, filename in enumerate(self._filenames):
            # Read in image and convert to grayscale
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img_shape is None:
                img_shape = gray.shape

            # Find the corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If the corners are found, add to the list of points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                self._num_images_passed += 1
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                if debug:
                    plt.subplot(m, m, count)
                    count += 1
                    plt.imshow(img[..., ::-1])
                    _, f = os.path.split(filename)
                    plt.title('Image #{}: {}'.format(i + 1, f))
                    plt.axis('off')

        # Now calibrate
        ret, mtx, dist, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape[::-1], None, None)
        if ret:
            self.K = mtx
            self.dist = dist
        else:
            print('Calibration was unfortunately unsuccessful')
            return False

        if debug:
            plt.show()

        return True

    def debug_undistorted_checkerboard(self):
        """Debug method to undistort the checkerboard images given the calibration parameters
        """

        n = len(self._filenames)
        m = int(np.sqrt(n))
        if n != m ** 2:
            m += 1
        plt.figure(figsize=(16, 12))

        for i, filename in enumerate(self._filenames):
            # Read in image and convert to grayscale
            img = cv2.imread(filename)
            dst = self.undistort(img)
            plt.subplot(m, m, i + 1)
            plt.imshow(dst[..., ::-1])
            _, f = os.path.split(filename)
            plt.title('Image #{}: {}'.format(i + 1, f))
            plt.axis('off')

        plt.show()

    def undistort_image(self, img):
        """Undistort an input image given the calibration parameters

        Args:
            img (numpy.ndarray): An image stored as a NumPy array

        Returns:
            Another numpy.ndarray illustrating the undistorted image
        """
        dst = cv2.undistort(img, self.K, self.dist, None, None)
        return dst

    def __repr__(self):
        """Pretty print the object state"""
        s = "Intrinsic Matrix:\n"
        s += str(self.K)
        s += "\nDistortion Coefficients: " + str(self.dist)
        s += "\nImages successful in finding checkerboard corners: {}/{}".format(
            self._num_images_passed, len(self._filenames))
        return s

    def __str__(self):
        """Allow for easy printing"""
        return self.__repr__()
