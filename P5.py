import os
import numpy as np
from moviepy.editor import VideoFileClip

from carndlib.camera import Calibrator
from carndlib.transform import Transformer
from carndlib.threshold import Thresholder
from carndlib.lines import LineFinder

from carndlib.cars import CarFinder
from carndlib.train import SVMTrainer


import carndlib.utils as utils

#  GLOBAL VARIABLES!
YSTART = 400
YSTOP = 656
SCALES = [1, 1.5, 2.0]
COLOR_SPACE = 'YCrCb'
SPATIAL_SIZE = (32, 32)
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HIST_BINS = 32
HIST_RANGE = (0, 256)
HOG_CHANNEL = 'ALL'

class Processor(object):
    """
        Image processor utility.
    """
    def __init__(self, trans, lines, svc, scaler):
        """
            Returns an image processor utility.
        """
        self.thresh = Thresholder()
        self.trans = trans
        self.lines = lines
        self.finder = CarFinder()
        self.svc = svc
        self.scaler = scaler

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
        result = self.finder.find_cars(img, COLOR_SPACE, YSTART, YSTOP, SCALES, self.svc, \
            self.scaler, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS, \
            False)
        if name is not None:
            utils.write_image(result, name)
        return self.finder.draw_detections(img)

    def process(self, img, name=None):
        """
            Processes an image.
        """
        result = self.find_cars(img, name)
        result = self.find_lane(result)
        return result

def pipeline():
    """
        Detects the lines on video imput.
    """
    cal = Calibrator()
    ret, mtx, dist, rvecs, tvecs = cal.calibrate()
    trans = Transformer(mtx, dist)
    lines = LineFinder(trans)
    trainer = SVMTrainer(color_space=COLOR_SPACE)
    svc, scaler = trainer.train()
    proc = Processor(trans, lines, svc, scaler)
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
    # lines = LineFinder(trans)
    trainer = SVMTrainer(color_space=COLOR_SPACE)
    svc, scaler = trainer.train()
    proc = Processor(None, None, svc, scaler)
    path = './test_images'
    out_path = './output_images'
    for img_name in utils.list_dir(path):
        base_path, name = os.path.split(img_name)
        print('Processing ' + name + '...')
        img = utils.read_image(img_name)
        result = proc.process(img, os.path.join(out_path, 'mix_' + name))
        utils.write_image(result, os.path.join(out_path, 'result_' + name))

if __name__ == "__main__":
    # test_pipeline()
    pipeline()

