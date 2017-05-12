import cv2
import numpy as np

import carndlib.utils as utils

class Thresholder(object):
    """
        An image thresholder utility.
    """

    def __init__(self):
        """
            Returns a Thresholder object.
        """
    
    def threshold(self, input, thresh=(0, 255)):
        """
            Applies the specified threshold to an image.
        """
        output = np.zeros_like(input)
        output[(input >= thresh[0]) & (input <= thresh[1])] = 1
        return output

    def gray_threshold(self, img, thresh=(130, 255)):
        """
            Grayscales and threshold an image.

            Attributes:
                img: the image.
                thresh: threshold lower and upper bounds tuple.
        """
        gray = utils.grayscale(img)
        return self.threshold(gray, thresh)

    def sobel(self, img, orient='x', kernel=3, thresh=(20, 100)):
        """
            Applies sobel filter in the *orient* direction.

            Attributes:
                img: the image.
                orient: orientation for to apply the filter.
                thresh: threshold lower and upper bounds tuple.
        """
        gray = utils.grayscale(img)
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
        # Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        return self.threshold(scaled_sobel, thresh)

    def grad_magnitude(self, img, kernel=3, thresh=(30, 100)):
        """
            Applies sobel on both orientations (x and y) and then
            computes the gradient magnitude and applies a threshold.

            Attributes:
                img: the image.
                kernel: The sobel kernel.
                thresh: threshold lower and upper bounds tuple.
        """
        gray = utils.grayscale(img)
        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
        # Calculate the magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
       #  Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        return self.threshold(gradmag, thresh)

    def direction(self, img, kernel=3, thresh=(0.7, 1.3)):
        """
            Applies sobel x and y, computes the direction of the
            gradient and then applies a threshold.

            Attributes:
                img: the image.
                kernel: The sobel kernel.
                thresh: threshold lower and upper bounds tuple.
        """
        gray = utils.grayscale(img)
        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
        # Take the absolute value of the x and y gradients
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        return self.threshold(absgraddir, thresh)

    def combine_two(self, first, second, tuples=False):
        """
            Applies a combination of two or four thresholds.

            Attributes:
                first: first threshold or tuple.
                second: second threshold or tuple.
                tuples: True if first and second are tuples
        """
        if tuples:
            combined = np.zeros_like(second[1])
            combined[((first[0] == 1) & (first[1] == 1)) | \
                ((second[0] == 1) & (second[1] == 1))] = 1
        else:
            combined = np.zeros_like(second)
            combined[(first == 1) | (second == 1)] = 1
        return combined

    def combine_all(self, img, ksize=3):
        """
            Computes and combines all thresholds.

            Attributes:
                img: the image.
                ksize: kernel size for thresholds.
        """
        sobelx = self.sobel(img, orient='x', kernel=ksize) #, thresh=(0, 255))
        sobely = self.sobel(img, orient='y', kernel=ksize) #, thresh=(0, 255))
        gradmag = self.grad_magnitude(img, kernel=ksize) #, thresh=(0, 255))
        dir_thresh = self.direction(img, kernel=ksize) #, thresh=(0, np.pi/2))
        return self.combine_two((sobelx, sobely), (gradmag, dir_thresh), True)

