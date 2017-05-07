import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

import VehicleDetection.utils as cutils

class SVCTrainer(object):
    """
        A SVC trainer utility.
    """

    def __init__(self):
        """
            Returns a SVCTrainer object.
        """

    def split_sets(self, cars_feats, notcars_feats, scaled_X):
        """
            Splits data into training and tests sets:
            X_train, X_test, y_train, y_test
        """
        y = np.hstack((np.ones(len(cars_feats)), np.zeros(len(notcars_feats))))
        rand_state = np.random.randint(0, 100)
        return train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    def get_scaler(self, cars_feats, notcars_feats):
        """
            Fits a StandardScaler.
        """
        X = np.vstack((cars_feats, notcars_feats)).astype(np.float64)
        print('X features shape', X.shape)
        if os.path.isfile('./xscaler.pkl'):
            X_scaler = cutils.load_scaler()
        else:
            X_scaler = StandardScaler().fit(X)
            cutils.save_scaler(X_scaler)
        scaled_X = X_scaler.transform(X)
        print("Scaled features: {}".format(len(scaled_X)))
        return X_scaler, scaled_X

    def train_svc(self, X_train, y_train, X_test, y_test, orient, pix_per_cell, cell_per_block):
        """
            Trains a SVC classifier
        """
        print("Using: {} orientation, {} pixels per cell and {} cells per block".format(orient, pix_per_cell, cell_per_block))
        print("Feature vector length: {}".format(len(X_train[0])))
        init = time.time()
        if os.path.isfile('./model.pkl'):
            svc = cutils.load_model()
        else:
            svc = SVC(C=1.0, probability=True)
            svc.fit(X_train, y_train)
            cutils.save_model(svc)
        end = time.time()
        print("SVC trained/loaded in {} seconds".format(round(end - init, 2)))
        print("Accuracy of SVC: {}".format(svc.score(X_test, y_test)))
        return svc
