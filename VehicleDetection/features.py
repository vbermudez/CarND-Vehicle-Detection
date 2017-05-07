from skimage.feature import hog
import numpy as np
import cv2
import os

import VehicleDetection.utils as cutils

class FeatureExtractor(object):
    """
        A features extractor utility.
    """

    def __init__(self):
        """
            Returns a FeatureExtractor object.
        """

    def get_hog_features(self, img, orient=9, pix_per_cell=8, \
        cell_per_block=2, vis=False, feature_vec=True):
        """
            Returns HOG features and visualization.
        """
        if vis:
            features, hog_image = hog(img, orientations=orient, \
                pixels_per_cell=(pix_per_cell, pix_per_cell), \
                cells_per_block=(cell_per_block, cell_per_block), \
                transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        else:
            features = hog(img, orientations=orient, \
                pixels_per_cell=(pix_per_cell, pix_per_cell), \
                cells_per_block=(cell_per_block, cell_per_block), \
                transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
            return features

    def bin_spatial(self, img, size=(32, 32)):
        """
            Returns a feature vector.
        """
        features = cv2.resize(img, size).ravel()
        return features

    def extract_features(self, imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, \
        hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, \
        spatial_feat=True, hist_feat=True, hog_feat=True):
        """
            Extracts features from a list of images
        """
        features = []
        for file in imgs:
            file_features = []
            image = cutils.read_image(file)
            if color_space != 'RGB':
                if color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else: feature_image = np.copy(image)
            if spatial_feat:
                spatial_features = self.bin_spatial(feature_image, spatial_size)
                file_features.append(spatial_features)
            if hist_feat:
                hist_features = cutils.color_hist(feature_image, hist_bins, hist_range)
                file_features.append(hist_features)
            if hog_feat:
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(self.get_hog_features(feature_image[:, :, channel], \
                            orient, pix_per_cell, cell_per_block, False, True))
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = self.get_hog_features(feature_image[:, :, hog_channel], \
                        orient, pix_per_cell, cell_per_block, False, True)
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        return features

    def single_img_features(self, img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, \
        hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, \
        spatial_feat=True, hist_feat=True, hog_feat=True, name=None):
        """
            Extracts features from a single image window.
        """
        img_features = []
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img)
        if spatial_feat:
            spatial_features = self.bin_spatial(feature_image, spatial_size)
            img_features.append(spatial_features)
        if hist_feat:
            hist_features = cutils.color_hist(feature_image, hist_bins, hist_range)
            img_features.append(hist_features)
        if hog_feat:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(self.get_hog_features(feature_image[:, :, channel], \
                            orient, pix_per_cell, cell_per_block, False, True))
            else:
                hog_features = self.get_hog_features(feature_image[:, :, hog_channel], \
                        orient, pix_per_cell, cell_per_block, False, True)
            img_features.append(hog_features)
        return np.concatenate(img_features)

    def img_features(self, cars, notcars, color_space='RGB', spatial_size=(32, 32), hist_bins=32, \
        hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, \
        spatial_feat=True, hist_feat=True, hog_feat=True):
        """
            Extract features from the images.
        """
        if os.path.isfile('./features.pkl'):
            cars_feats, notcars_feats = cutils.load_features()
        else:
            cars_feats = self.extract_features(cars, color_space, spatial_size, hist_bins, \
                hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, \
                hist_feat, hog_feat)
            notcars_feats = self.extract_features(notcars, color_space, spatial_size, hist_bins, \
                hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, \
                hist_feat, hog_feat)
            cutils.save_features(cars_feats, notcars_feats)
        print("Car features: {}.".format(len(cars_feats)))
        print("Not car features: {}.".format(len(notcars_feats)))
        return cars_feats, notcars_feats
        