import cv2
import numpy as np

import AdvLaneFinding.utils as utils

class Transformer(object):
    """
        An images processor utility.

        Attributes:
            mtx: Computed camera matrix
            dist: Distortion coefficients
    """

    def __init__(self, mtx=None, dist=None):
        """
            Returns a Transformer object with computed camera matrix *mtx*
            and distortion coefficients *dist*.
        """
        self.mtx = mtx
        self.dist = dist

    def undistort(self, img):
        """
            Undistorts an image

            Attributes:
                img: the image
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    
    def get_matrices(self):
        """
            Returns the perspective matrices.
        """
        # src = np.float32([[595.0, 450.0], [259.0, 687.0], [1056.0, 687.0], [687.0, 450.0]])
        # dst = np.float32([[259.0, 0.0], [259.0, 720.0], [1056.0, 720.0], [1056.0, 0.0]])
        src = np.float32([[1030, 670], [712, 468], [570, 468], [270, 670]])
        dst = np.float32([[1010, 720], [1010, 0], [280, 0], [280, 720]])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def warp(self, img, grayscale=False):
        """
            Warps an image

            Attributes:
                img: grayscaled image
        """
        if grayscale:
            gray = utils.grayscale(img)
        else:
            gray = img
        img_size = (gray.shape[1], gray.shape[0])
        M, Minv = self.get_matrices()
        return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    def unwarp(self, img, grayscale=False):
        """
            Unwarps an image
        """
        if grayscale:
            gray = utils.grayscale(img)
        else:
            gray = img
        img_size = (gray.shape[1], gray.shape[0])
        M, Minv = self.get_matrices()
        return cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)

