import numpy as np
import cv2
from scipy.ndimage.measurements import label

from VehicleDetection.features import FeatureExtractor
import VehicleDetection.utils as cutils

class CarFinder(object):
    """
        A car finder utility.
    """

    def __init__(self, feat):
        """
            Returns a CarFinder object.s
        """
        self.feat = feat

    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None], \
        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        """
            Returns a list of sliding windows.
        """
        if x_start_stop[0] is None:
            x_start_stop[0] = 0
        if x_start_stop[1] is None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] is None:
            y_start_stop[0] = 0
        if y_start_stop[1] is None:
            y_start_stop[1] = img.shape[0]
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        nx_windows = np.int(xspan / nx_pix_per_step) - 1
        ny_windows = np.int(yspan / ny_pix_per_step) - 1
        window_list = []
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                window_list.append(((startx, starty), (endx, endy)))
        return window_list

    def search_windows(self, img, windows, clf, scaler, color_space='RGB', \
        spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), orient=9, \
        pix_per_cell=8, cell_per_block=2, hog_channel=0, \
        spatial_feat=True, hist_feat=True, hog_feat=True):
        """
            Searches into an image usgin a list of windows.
        """
        on_windows = []
        for window in windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], \
                window[0][0]:window[1][0]], (64, 64))
            features = self.feat.single_img_features(test_img, color_space, \
                spatial_size, hist_bins, hist_range, orient, pix_per_cell, \
                cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            prediction = clf.predict(test_features)
            if prediction == 1:
                on_windows.append(window)
        return on_windows

    def find_cars(self, image, ystart, ystop, scale, svc, X_scaler, orient, \
        pix_per_cell, cell_per_block, spatial_size, hist_bins, \
        hist_range, window=64, name=None):
        """
            Finds cars into an image
        """
        samples = name is not None
        img = image.astype(np.float32) / 255
        img_tosearch = img[ystart:ystop, :, :]
        if samples:
            cutils.write_two_img([img, img_tosearch], ['Original', 'Search Area RGB'], \
                'search_area_rgb_' + name)
        ctrans_tosearch = cutils.convert_color(img_tosearch, conv='RGB2YCrCb')
        if samples:
            cutils.write_two_img([img, ctrans_tosearch], ['Original', 'Search Area YCrCb'], \
                'search_area_ycrcb_' + name)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, \
                (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
            if samples:
                cutils.write_two_img([img, ctrans_tosearch], ['Original', 'Scaled'], \
                    'scaled_' + name)
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        # window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        # Compute individual channel HOG features for the entire image
        hog1, hog1_img = self.feat.get_hog_features(ch1, orient, pix_per_cell, \
            cell_per_block, vis=True, feature_vec=False)
        hog2, hog2_img = self.feat.get_hog_features(ch2, orient, pix_per_cell, \
            cell_per_block, vis=True, feature_vec=False)
        hog3, hog3_img = self.feat.get_hog_features(ch3, orient, pix_per_cell, \
            cell_per_block, vis=True, feature_vec=False)
        if samples:
            cutils.write_two_img([ctrans_tosearch, hog1_img], \
                ['Search Area YCrCb', 'HOG (red channel)'], 'red_hog_' + name)
            cutils.write_two_img([ctrans_tosearch, hog2_img], \
                ['Search Area YCrCb', 'HOG (green channel)'], 'green_hog_' + name)
            cutils.write_two_img([ctrans_tosearch, hog3_img], \
                ['Search Area YCrCb', 'HOG (blue channel)'], 'blue_hog_' + name)
        bboxes = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))
                # Get color features
                spatial_features = self.feat.bin_spatial(subimg, spatial_size)
                hist_features = cutils.color_hist(subimg, hist_bins, hist_range)
                # Scale features and make a prediction
                all_features = np.hstack((spatial_features, hist_features, \
                    hog_features)).reshape(1, -1)
                # print('Shapes:', spatial_features.shape, hist_features.shape, hog_features.shape, all_features.shape)
                test_features = X_scaler.transform(all_features)
                #test_features = \
                #   X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    if samples:
                        pred_img = np.copy(img)
                        cv2.rectangle(pred_img, (xbox_left, ytop_draw + ystart), \
                            (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                        cutils.write_two_img([img, pred_img], ['Original', 'Prediction'], \
                            'blue_hog_' + name)
                    bboxes.append(((xbox_left, ytop_draw + ystart), \
                        (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
        # return draw_img
        # return bboxes
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = cutils.add_heat(heat, bboxes)
        heat = cutils.apply_threshold(heat, 1)
        heatmap = np.clip(heat, 0, 255)
        if samples:
            cutils.write_two_img([img, heatmap], ['Original', 'Heatmap'], 'heatmap_' + name, cmap2='hot')
        labels = label(heatmap)
        if samples:
            cutils.write_two_img([img, labels], ['Original', 'Labels'], 'labels_' + name, cmap2='gray')
        draw_img = np.copy(img)
        draw_img = cutils.draw_labeled_bboxes(draw_img, labels)
        return draw_img
        