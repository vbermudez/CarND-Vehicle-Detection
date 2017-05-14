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

def sample_frame(image, finder, svc, scaler):
    """
        Extract samples from the viedo.
    """
    from scipy.ndimage.measurements import label
    import time
    img = np.copy(image)
    name = 'heatmap_' + str(time.time()) + '.jpg'
    acc_name = 'acc_heatmap_' + str(time.time()) + '.jpg'
    bbox_name = 'bbox_' + str(time.time()) + '.jpg'
    bbox_list = \
            finder.find_bboxes(image, COLOR_SPACE, YSTART, YSTOP, SCALES, svc, \
            scaler, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, \
            HIST_BINS)
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = finder.add_heat(heat, bbox_list)
    heat = finder.apply_threshold(heat, 1)
    hm = np.clip(heat, 0, 255)
    utils.plot_one_image(hm, 'Heatmap', name, cmap='hot')
    heatmap = finder.heatmap(image, bbox_list)
    utils.plot_one_image(heatmap, 'Accumulative heatmap', acc_name, cmap='hot')
    labels = label(heatmap)
    result = finder.draw_labeled_boxes(np.copy(image), labels)
    utils.plot_one_image(result, 'Bounding box', bbox_name)
    return result

def samples_pipeline():
    """
        Generate sample for writeup
    """
    # from scipy.ndimage.measurements import label
    # from carndlib.features import Features
    # feats = Features()
    # print('Sampling features...')
    # for img_name in utils.list_dir('./features'):
    #     base_path, name = os.path.split(img_name)
    #     print('Sampling ' + name + '...')
    #     image = utils.read_image(img_name)
    #     if name.startswith('car'):
    #         title = 'Car'
    #     else:
    #         title = 'Not car'
    #     utils.plot_one_image(image, title, name)
    #     ch1 = image[:, :, 0]
    #     ch2 = image[:, :, 1]
    #     ch3 = image[:, :, 2]
    #     hog, output = feats.get_hog(ch1, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, vis=True)
    #     utils.write_two_img([ch1, output], ['Original CH-1', 'HOG CH-1'], 'hog1_' + name)
    #     hog, output = feats.get_hog(ch2, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, vis=True)
    #     utils.write_two_img([ch2, output], ['Original CH-2', 'HOG CH-2'], 'hog2_' + name)
    #     hog, output = feats.get_hog(ch3, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, vis=True)
    #     utils.write_two_img([ch3, output], ['Original CH-3', 'HOG CH-3'], 'hog3_' + name)
    #     output = feats.bin_spatial(image, size=SPATIAL_SIZE)
    #     utils.plot_histogram(image, output, 'spatial_' + name, title='Color bin')
    #     output = feats.color_hist(image, nbins=HIST_BINS)
    #     utils.plot_histogram(image, output, 'histogram_'+ name)
    finder = CarFinder()
    svc = utils.load('./' + COLOR_SPACE + '_model.pkl')
    scaler = utils.load('./' + COLOR_SPACE + '_scaler.pkl')
    # print('Samplig process...')
    # for img_name in utils.list_dir('./test_images'):
    #     base_path, name = os.path.split(img_name)
    #     print('Sampling ' + name + '...')
    #     image = utils.read_image(img_name)
        # for scale in SCALES:
        #     output, bboxes = finder.scan_image(image, COLOR_SPACE, YSTART, \
        #         YSTOP, 0, image.shape[1], scale, svc, scaler, ORIENT, \
        #         PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS, vis=True)
        #     utils.plot_one_image(output, 'Sliding Window, scale: {}'.format(scale), \
        #         'sliding_win_' + str(scale) + name)
    #     output, out_imgs, bboxes = finder.find_bboxes(image, COLOR_SPACE, YSTART, \
    #         YSTOP, SCALES, svc, scaler, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, \
    #         SPATIAL_SIZE, HIST_BINS, vis=True)
    #     utils.plot_one_image(output, 'Detections', 'detections_' + '_' + name)
    #     heatmap = finder.heatmap(image, bboxes)
    #     utils.plot_one_image(heatmap, 'Heatmap', 'heatmap_' + name, cmap='hot')
    #     labels = label(heatmap)
    #     output = finder.draw_labeled_boxes(np.copy(image), labels)
    #     utils.plot_one_image(output, 'Contour', 'contour_' + name)
    #     output = finder.draw_detections(image)
    #     utils.plot_one_image(output, 'Final', 'result_' + name)
    video = VideoFileClip('./project_videoTrim.mp4')
    output = video.fl_image(lambda image: sample_frame(image, finder, svc, scaler))
    output.write_videofile('./outputTrim.mp4', audio=False)

if __name__ == "__main__":
    # test_pipeline()
    pipeline()
    # samples_pipeline()

