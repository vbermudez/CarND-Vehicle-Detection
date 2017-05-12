from skimage.feature import hog
import numpy as np
import cv2
import os

import carndlib.utils as utils

class Features(object):
    """
        A features extractor utility.
    """

    def __init__(self):
        """
            Returns a Features object.
        """

    def get_hog(self, img, orient, pix_per_cell, cell_per_block, \
        vis=False, feature_vec=True):
        """
            Returns HOG features from a image.
        """
        if vis:
            features, hog_image = hog(img, orientations=orient, \
                                      pixels_per_cell=(pix_per_cell, pix_per_cell), \
                                      cells_per_block=(cell_per_block, cell_per_block), \
                                      transform_sqrt=False, \
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        else:
            features = hog(img, orientations=orient, \
                           pixels_per_cell=(pix_per_cell, pix_per_cell), \
                           cells_per_block=(cell_per_block, cell_per_block), \
                           transform_sqrt=False, \
                           visualise=vis, feature_vector=feature_vec)
            return features

    def bin_spatial(self, img, size=(32, 32)):
        """
            Returns a feature vector.
        """
        clr1 = cv2.resize(img[:, :, 0], size).ravel()
        clr2 = cv2.resize(img[:, :, 1], size).ravel()
        clr3 = cv2.resize(img[:, :, 2], size).ravel()
        return np.hstack((clr1, clr2, clr3))

    def color_hist(self, img, nbins=32):
        """
            Computes the histogram of the color channels separately.
        """
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        return hist_features

    def extract_all(self, imgs, color_space='RGB', spatial_size=(32, 32), \
        hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, \
        spatial_feat=True, hist_feat=True, hog_feat=True):
        """
            Extracts features from a list of images
        """
        features = []
        for f in imgs:
            file_features = []
            image = utils.read_image(f)
            feature_image = utils.to_color_space(image, color_space)
            if spatial_feat:
                spatial_features = self.bin_spatial(feature_image, size=spatial_size)
                file_features.append(spatial_features)
            if hist_feat:
                hist_features = self.color_hist(feature_image, nbins=hist_bins)
                file_features.append(hist_features)
            if hog_feat:
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(self.get_hog(feature_image[:, :, channel], \
                            orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = self.get_hog(feature_image[:, :, hog_channel], orient, \
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        return features
        