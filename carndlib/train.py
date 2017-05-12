import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob

import carndlib.utils as utils
from carndlib.features import Features

class SVMTrainer(object):
    """
        A SVM trainer utility.
    """

    def __init__(self, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, \
        pix_per_cell=8, cell_per_block=2, hog_channel='ALL', spatial_feat=True, hist_feat=True, \
        hog_feat=True):
        """
            Returns a SVMTrainer object.
        """
        self.cars = None
        self.notcars = None
        self.img_features = None
        self.labels = None
        self.scaler = None
        self.svc = None
        self.features_file = './' + color_space + '_feats.pkl'
        self.scaler_file = './' + color_space + '_scaler.pkl'
        self.model_file = './' + color_space + '_model.pkl'
        self.features = Features()
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat

    def save_features(self):
        """
            Saves the features to a file.
        """
        data = {"features": self.img_features, "labels": self.labels}
        utils.save(data, self.features_file)

    def load_features(self):
        """
            Loads the features from a file.
        """
        data = utils.load(self.features_file)
        self.img_features = data["features"]
        self.labels = data["labels"]

    def save_scaler(self):
        """
            Saves the scaler to a file.
        """
        utils.save(self.scaler, self.scaler_file)

    def load_scaler(self):
        """
            Loads the scaler from a file.
        """
        self.scaler = utils.load(self.scaler_file)

    def save_model(self):
        """
            Saves the trained model to a file.
        """
        utils.save(self.svc, self.model_file)

    def load_model(self):
        """
            Loads the trained model from a file.
        """
        self.svc = utils.load(self.model_file)

    def load_samples(self):
        """
            Loads the cars and not-cars images.
        """
        print("Loading samples...")
        self.cars = glob.glob("./vehicles/*/*.png")
        self.notcars = glob.glob("./non-vehicles/*/*.png")
        print("Car samples: {}.".format(len(self.cars)))
        print("Not car samples: {}.".format(len(self.notcars)))

    def scale_features(self, feats):
        """
            Fits the scaler if not exists, and applies it to the features.
        """
        if utils.file_exists(self.scaler_file):
            self.load_scaler()
            print("Scaler loaded.")
        else:
            self.scaler = StandardScaler().fit(feats)
            self.save_scaler()
            print("Scaler saved.")
        print("Scaling features...")
        self.img_features = self.scaler.transform(feats)
        print("Features scaled.")

    def extract_features(self):
        """
            Gets the features from all the samples.
        """
        if utils.file_exists(self.features_file):
            print("Loading features...")
            self.load_features()
            print("Features loaded.")
        else:
            print("Extracting features...")
            cars_feats = self.features.extract_all(self.cars, self.color_space, \
                self.spatial_size, self.hist_bins, self.orient, self.pix_per_cell, \
                self.cell_per_block, self.hog_channel, self.spatial_feat, \
                self.hist_feat, self.hog_feat)
            notcars_feats = self.features.extract_all(self.notcars, self.color_space, \
                self.spatial_size, self.hist_bins, self.orient, self.pix_per_cell, \
                self.cell_per_block, self.hog_channel, self.spatial_feat, \
                self.hist_feat, self.hog_feat)
            print("Car feats.: {} x {}.".format(len(cars_feats), len(cars_feats[0])))
            print("Not car feats.: {} x {}.".format(len(notcars_feats), len(notcars_feats[0])))
            feats = np.vstack((cars_feats, notcars_feats)).astype(np.float64)
            self.scale_features(feats)
            self.labels = np.hstack((np.ones(len(cars_feats)), np.zeros(len(notcars_feats))))
            self.save_features()
            print("Features saved.")

    def split_sets(self):
        """
            Splits data into training and tests sets:
            X_train, X_test, y_train, y_test
        """
        rand_state = np.random.randint(0, 100)
        return train_test_split(self.img_features, self.labels, test_size=0.2, random_state=rand_state)

    def train(self):
        """
            Trains a SVM classifier adn returns the scaler and the trained SVM.
        """
        print("Using: {} orientation, {} pixels per cell and {} cells per block".format(\
            self.orient, self.pix_per_cell, self.cell_per_block))
        init = time.time()
        if utils.file_exists(self.model_file):
            if self.img_features is None:
                print("Loading features...")
                self.load_features()
                print("Features loaded.")
            if self.scaler is None:
                print("Loading scaler...")
                self.load_scaler()
                print("Scaler loaded.")
            X_train, X_test, y_train, y_test = self.split_sets()
            print("Loading model...")
            self.load_model()
            print("Model loaded.")
        else:
            self.load_samples()
            self.extract_features()
            X_train, X_test, y_train, y_test = self.split_sets()
            print("Feature vector length: {}".format(len(X_train[0])))
            self.svc = SVC(C=1.0, probability=True)
            print("Training model...")
            self.svc.fit(X_train, y_train)
            self.save_model()
            print("Model saved.")
        end = time.time()
        print("SVC trained/loaded in {} minutes".format(round((end - init) / 60, 2)))
        print("Accuracy of SVC: {}".format(self.svc.score(X_test, y_test)))
        return self.svc, self.scaler
