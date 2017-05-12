import numpy as np
import cv2
from scipy.ndimage.measurements import label

from carndlib.features import Features
import carndlib.utils as utils

class CarFinder(object):
    """
        A car finder utility.
    """

    def __init__(self):
        """
            Returns a CarFinder object.
        """
        self.features = Features()
        self.heatmaps = []
        self.aggregate_heatmap = np.zeros((720, 1280)).astype(np.float64)
        self.max_heatmaps = 10
        self.num_cars = 0
        self.boxes_finded = [] 

    def scan_image(self, img, color_space, ystart, ystop, xstart, xstop, scale, \
        svc, scaler, orient, pix_per_cell, cell_per_block, spatial_size, \
        hist_bins, vis=False):
        """
            Scans the image and create bounding boxes for any detection.
        """
        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255
        img_tosearch = img[ystart:ystop, xstart:xstop, :]
        ctrans_tosearch = utils.to_color_space(img_tosearch, color_space)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, \
                (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - 1
        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        # Compute individual channel HOG features for the entire image
        hog1 = self.features.get_hog(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = self.features.get_hog(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = self.features.get_hog(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        bbox_list = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = \
                    hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = \
                    hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = \
                    hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell
                # Extract the image patch
                subimg = \
                    cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
                # Get color features
                spatial_features = self.features.bin_spatial(subimg, size=spatial_size)
                hist_features = self.features.color_hist(subimg, nbins=hist_bins)
                # Scale features and make a prediction
                test_features = scaler.transform(np.hstack((spatial_features, hist_features, \
                    hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)
                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    bbox_list.append(((xbox_left + xstart, ytop_draw + ystart), \
                        (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))
                if vis:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(draw_img, (xbox_left + xstart, ytop_draw + ystart), \
                        (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart), \
                        (255, 0, 0), 2)
        if vis:
            return draw_img, bbox_list
        else:
            return bbox_list

    def find_bboxes(self, img, color_space, ystart, ystop, scales, \
        svc, scaler, orient, pix_per_cell, cell_per_block, spatial_size, \
        hist_bins, vis=False):
        """
            Finds all the bounding boxes in the image over usgin a pyramidal search.
        """
        xstart = img.shape[1] // 2
        xstop = img.shape[1]
        bbox_list = []
        d_images = []
        for scale in scales:
            if vis:
                d_img, bboxes = self.scan_image(img, color_space, \
                    ystart, ystop, xstart, xstop, scale, svc, scaler, \
                    orient, pix_per_cell, cell_per_block, spatial_size, \
                    hist_bins, vis)
                d_images.append(d_img)
            else:
                bboxes = self.scan_image(img, color_space, \
                    ystart, ystop, xstart, xstop, scale, svc, scaler, \
                    orient, pix_per_cell, cell_per_block, spatial_size, \
                    hist_bins, vis)
            bbox_list.extend(bboxes)
        if vis:
            draw_img = np.copy(img)
            for bbox in bbox_list:
                cv2.rectangle(draw_img, (bbox[0][0], bbox[0][1]), \
                    (bbox[1][0], bbox[1][1]), (0, 0, 255), 6)
            return draw_img, d_images, bbox_list
        else:
            return bbox_list

    def add_heat(self, heatmap, bbox_list):
        """
            Adds "heat" to a list of bounding boxes.
        """
        for box in bbox_list:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heatmap

    def apply_threshold(self, heatmap, threshold):
        """
            Applies a threshold on a heatmap.
        """
        heatmap[heatmap <= threshold] = 0
        return heatmap

    def heatmap(self, img, bbox_list, threshold=1):
        """
            Computes the accumulative heatmap of bounding boxes.
        """
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        # Add heat to each box in box list
        heat = self.add_heat(heat, bbox_list)
        heat = self.apply_threshold(heat, threshold)
        heatmap = np.clip(heat, 0, 255)
        self.heatmaps.append(heatmap)
        self.aggregate_heatmap = self.aggregate_heatmap + heatmap
        if len(self.heatmaps) > self.max_heatmaps:
            self.aggregate_heatmap = self.aggregate_heatmap - self.heatmaps.pop(0)
            self.aggregate_heatmap = np.clip(self.aggregate_heatmap, 0.0, 9999999.0)
        return self.aggregate_heatmap

    def draw_labeled_boxes(self, img, labels, color=(255, 0, 0), thick=3):
        """
            Draws a box around the detected hot spot.
        """
        self.num_cars = labels[1]
        self.boxes_finded = []
        for car in range(1, labels[1] + 1):
            nonzero = (labels[0] == car).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(img, bbox[0], bbox[1], color, thick)
            self.boxes_finded.append(bbox)
        return img

    def find_cars(self, img, color_space, ystart, ystop, scales, \
        svc, scaler, orient, pix_per_cell, cell_per_block, spatial_size, \
        hist_bins, vis=False):
        """
            Finds cars in the image.
        """
        bbox_list = \
            self.find_bboxes(img, color_space, ystart, ystop, scales, svc, \
            scaler, orient, pix_per_cell, cell_per_block, spatial_size, \
            hist_bins, vis)
        heatmap = self.heatmap(img, bbox_list)
        labels = label(heatmap)
        return self.draw_labeled_boxes(np.copy(img), labels)

    def draw_detections(self, img, color=(255, 255, 0), alpha=.3):
        """
            Draws the bounding box over detected car.
        """
        overlay = img.copy()
        for bbox in self.boxes_finded:
            cv2.rectangle(overlay, bbox[0], bbox[1], color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1- alpha, 0, img)
        text = "Cars finded {}".format(self.num_cars)
        cv2.putText(img, text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return img


