import os
import numpy as np
from moviepy.editor import VideoFileClip

from AdvLaneFinding.camera import Calibrator
from AdvLaneFinding.transform import Transformer
from AdvLaneFinding.threshold import Thresholder
from AdvLaneFinding.lines import LineFinder
import AdvLaneFinding.utils as utils
from VehicleDetection.features import FeatureExtractor
from VehicleDetection.cars import CarFinder
from VehicleDetection.train import SVCTrainer
import VehicleDetection.utils as cutils

#  GLOBAL VARIABLES!
YSTART = 400
YSTOP = 656
SCALE = 1
COLOR_SPACE = 'RGB'
SPATIAL_SIZE = (32, 32)
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HIST_BINS = 32
HIST_RANGE = (0, 256)
HOG_CHANNEL = 'ALL'
WINDOW = 64

class Processor(object):
    """
        Image processor utility.
    """
    def __init__(self, thresh, trans, lines, feat, finder, svc, X_scaler):
        """
            Returns an image processor utility.
        """
        self.thresh = thresh
        self.trans = trans
        self.lines = lines
        self.feat = feat
        self.finder = finder
        self.svc = svc
        self.X_scaler = X_scaler

    def find_lane(self, img):
        """
            Finds the lane.
        """
        undist = self.trans.undistort(img)
        mag = self.thresh.grad_magnitude(undist, 5, (130, 255))
        equ = utils.equalize(undist)
        color_thresh = self.thresh.threshold(equ, (251, 255))
        combined = self.thresh.combine_two(mag, color_thresh)
        warped = self.trans.warp(combined)
        result = self.lines.find_lines(warped, img, False)
        return result

    def find_cars(self, img, name):
        """
            Fins the cars.
        """
        return self.finder.find_cars(img, YSTART, YSTOP, SCALE, self.svc, self.X_scaler, \
            ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS, \
            HIST_RANGE, WINDOW, name)

    def process(self, img, name=None):
        """
            Processes an image.
        """
        # lane = self.find_lane(img)
        result = self.find_cars(img, name) # lane
        return result

def train_pipeline(feat):
    """
        Trains a the SVC.
    """
    trainer = SVCTrainer()
    cars, notcars = cutils.load_images()
    cars_feats, notcars_feats = feat.img_features(cars, notcars, COLOR_SPACE, SPATIAL_SIZE, \
        HIST_BINS, HIST_RANGE, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, HOG_CHANNEL, True, True, True)
    X_scaler, scaled_X = trainer.get_scaler(cars_feats, notcars_feats)
    X_train, X_test, y_train, y_test = trainer.split_sets(cars_feats, notcars_feats, scaled_X)
    svc = trainer.train_svc(X_train, y_train, X_test, y_test, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK)
    return svc, X_scaler

def pipeline():
    """
        Detects the lines on video imput.
    """
    cal = Calibrator()
    ret, mtx, dist, rvecs, tvecs = cal.calibrate()
    trans = Transformer(mtx, dist)
    thresh = Thresholder()
    lines = LineFinder(trans)
    feat = FeatureExtractor()
    finder = CarFinder(feat)
    svc, X_scaler = train_pipeline(feat)
    proc = Processor(thresh, trans, lines, feat, finder, svc, X_scaler)
    video = VideoFileClip('./project_video.mp4')
    output = video.fl_image(proc.process)
    output.write_videofile('./output.mp4', audio=False)

def test_pipeline():
    """
        Tests the pipelin with static images
    """
    # cal = Calibrator()
    # ret, mtx, dist, rvecs, tvecs = cal.calibrate()
    # trans = Transformer(mtx, dist)
    # thresh = Thresholder()
    # lines = LineFinder(trans)
    feat = FeatureExtractor()
    finder = CarFinder(feat)
    svc, X_scaler = train_pipeline(feat)
    proc = Processor(None, None, None, feat, finder, svc, X_scaler)
    # proc = Processor(thresh, trans, lines, feat, finder, svc, X_scaler)
    path = './test_images'
    out_path = './output_images'
    for img_name in utils.list_dir(path):
        base_path, name = os.path.split(img_name)
        print('Processing ' + name + '...')
        img = utils.read_image(img_name)
        result = proc.process(img, name)
        utils.write_image(result, os.path.join(out_path, 'result_' + name))

if __name__ == "__main__":
    test_pipeline()
    # pipeline()

