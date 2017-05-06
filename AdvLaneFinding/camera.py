import cv2
import numpy as np

import AdvLaneFinding.utils as utils

class Calibrator(object):
    """
        A camera calibration utility.

        Attributes:
            source: Source directory containing calibration images.
            dest: Destination directy for sample undistorted and warped images.
            nx: The number of corners in x.
            ny: The number of corners in y.
    """

    def __init__(self, source='./camera_cal', nx=9, ny=6):
        """
            Returns a Calibrator object with source path *source*,
            corners in x *nx* and corners in y *ny*
        """
        self.source = source
        self.nx = nx
        self.ny = ny

    def find_corners(self, img):
        """
            Finds the chessboard corners of a image
        """
        gray = utils.grayscale(img)
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
        return ret, corners, gray

    def calibrate(self):
        """
            Finds the chessboard corners of the images
        """
        obj_p = np.zeros((self.ny * self.nx, 3), np.float32)
        obj_p[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
        obj_points = []
        img_points = []
        for img_name in utils.list_dir(self.source):
            img = utils.read_image(img_name)
            ret, corners, gray = self.find_corners(img)
            if ret:
                obj_points.append(obj_p)
                img_points.append(corners)
        return cv2.calibrateCamera(obj_points, img_points, gray.shape[0:2], None, None)
