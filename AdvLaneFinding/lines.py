import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks_cwt

import AdvLaneFinding.utils as utils

class LineFinder(object):
    """
        A lane line finder utility.
    """

    def __init__(self, transform):
        """
            Returns as LineFinder object.
        """
        self.left_line = Line()
        self.right_line = Line()
        self.ym_per_pix = 3*7/720 # updated as suggested in review
        self.xm_per_pix = 3.7/700 # base value, use get_xm_pr_pix for correct one
        self.transform = transform

    def histogram(self, img):
        """
            Returns the histogram of an image.

            Attributes:
                img: the image.
                plot_hist: specifies if the histogram must be plotted.
                output: where to write the plotted histogram.
                name: name of the plotted histogram.
        """
        return np.sum(img[img.shape[0] / 2:, :], axis=0)

    def get_xm_pr_pix(self, image_size, lane_fit):
        """
            Calculates the lane width.
        """
        left_fit, right_fit = lane_fit
        left_c = left_fit[0] * image_size[0] ** 2 + left_fit[1] * image_size[0] + left_fit[2]
        right_c = right_fit[0] * image_size[0] ** 2 + right_fit[1] * image_size[0] + right_fit[2]
        width = right_c - left_c
        self.xm_per_pix = 3.7 / width
        return self.xm_per_pix

    def find_lines_base(self, img):
        """
            Finds the coordinates of the base of the lane line (left, right).
            Mixing old and new colde.
        """
        histogram = self.histogram(img)
        indexes = find_peaks_cwt(histogram, np.arange(1, 550))
        base_inds = (indexes[0], indexes[-1])
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        return (leftx_base, rightx_base), base_inds

    def get_nonzero(self, img):
        """
            Gets the non-zero pixels of the image.
        """
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        return (nonzerox, nonzeroy)

    def find_complete_line(self, img, lane_base, nwindows=9, win_width=100, minpix=50):
        """
            Uses a sliding window technique to find all the pixels in that lane.
        """
        leftx_base, rightx_base = lane_base
        nonzerox, nonzeroy = self.get_nonzero(img)
        output = np.dstack((img, img, img)) * 255
        win_h = np.int(img.shape[0] / nwindows)
        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []
        if not self.left_line.detected:
            for window in range(nwindows):
                win_y_low = img.shape[0] - (window + 1) * win_h
                win_y_high = img.shape[0] - window * win_h
                win_xleft_low = leftx_current - win_width
                win_xleft_high = leftx_current + win_width
                win_xright_low = rightx_current - win_width
                win_xright_high = rightx_current + win_width
                cv2.rectangle(output, (win_xleft_low, win_y_low), \
                    (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(output, (win_xright_low, win_y_low), \
                    (win_xright_high, win_y_high), (0, 255, 0), 2)
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                    (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                    (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        else:
            left_lane_inds = ((nonzerox > (self.left_line.current_fit[0] * (nonzeroy**2) + \
                self.left_line.current_fit[1] * nonzeroy + \
                self.left_line.current_fit[2] - win_width)) & \
                (nonzerox < (self.left_line.current_fit[0]*(nonzeroy**2) + \
                self.left_line.current_fit[1] * nonzeroy + \
                self.left_line.current_fit[2] + win_width)))
            right_lane_inds = ((nonzerox > (self.right_line.current_fit[0] * (nonzeroy**2) + \
                self.right_line.current_fit[1] * nonzeroy + \
                self.right_line.current_fit[2] - win_width)) & \
                (nonzerox < (self.right_line.current_fit[0] * (nonzeroy**2) + \
                self.right_line.current_fit[1] * nonzeroy + \
                self.right_line.current_fit[2] + win_width)))
        return (nonzerox, nonzeroy), (left_lane_inds, right_lane_inds), output

    def get_pixels_pos(self, nonzero, lane_inds):
        """
            Gets the left and right pixel positions
        """
        nonzerox, nonzeroy = nonzero
        left_lane_inds, right_lane_inds = lane_inds
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        return (leftx, lefty), (rightx, righty)

    def get_curved_lines(self, img, left_pos, right_pos):
        """
            Calculates a 2nd order polynomial that fits the pixels
        """
        leftx, lefty = left_pos
        rightx, righty = right_pos
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        return ploty, (left_fit, right_fit), (left_fitx, right_fitx)

    def get_lines_curvature(self, img, ploty, left_pos, right_pos, lane_fit, lane_fitx):
        """
            Calculates the curvature of a line.
        """
        leftx, lefty = left_pos
        rightx, righty = right_pos
        left_fit, right_fit = lane_fit
        left_fitx, right_fitx = lane_fitx
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + \
            left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + \
            right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])
        self.left_line.was_detected(left_fitx, left_curverad, left_fit, right_curverad, right_fit)
        self.right_line.was_detected(right_fitx, right_curverad, right_fit, left_curverad, \
            left_fit, not self.left_line.detected)
        left_fit_cr = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * self.ym_per_pix, rightx * self.xm_per_pix, 2)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + \
            left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + \
            right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
        return (left_curverad, right_curverad)

    def get_car_position(self, img, base_inds):
        """
            Calculates the distance of the car from the center of the lane.
        """
        left_base, right_base = base_inds
        image_center = img.shape[1] / 2
        car_middle_pixel = (left_base + right_base) / 2
        return (car_middle_pixel - image_center) * self.xm_per_pix

    def draw_lines(self, img, original, ploty):
        """
            Draws the lane lines on the image
        """
        warp_zero = np.zeros_like(img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        left_avg_fitx = self.left_line.best_fit[0] * ploty**2 + self.left_line.best_fit[1] * \
            ploty + self.left_line.best_fit[2]
        right_avg_fitx = self.right_line.best_fit[0] * ploty**2 + self.right_line.best_fit[1] * \
            ploty + self.right_line.best_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_avg_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_avg_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.polylines(color_warp, np.int32([pts_left]), False, (255, 0, 0), 50)
        cv2.polylines(color_warp, np.int32([pts_right]), False, (0, 0, 255), 50)
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        newwarp = self.transform.unwarp(color_warp)
        return cv2.addWeighted(original, 1, newwarp, 0.3, 0)

    def find_lines(self, warped, original, debug=False):
        """
            Finds the lane lines. Code from lectures.

            Attributes:
                img: the image.
        """
        lane_base, base_inds = self.find_lines_base(warped)
        nonzero, lane_inds, sw_output = self.find_complete_line(warped, lane_base)
        left_pos, right_pos = self.get_pixels_pos(nonzero, lane_inds)
        ploty, lane_fit, lane_fitx = self.get_curved_lines(warped, left_pos, \
            right_pos)
        if debug:
            left_lane_inds, right_lane_inds = lane_inds
            nonzerox, nonzeroy = nonzero
            sw_output[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            sw_output[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        self.get_xm_pr_pix(warped.shape, lane_fit)
        curverad = self.get_lines_curvature(warped, ploty, left_pos, \
            right_pos, lane_fit, lane_fitx)
        car_offset = self.get_car_position(warped, base_inds)
        left_curverad, right_curverad = curverad
        self.left_line.set_output_params(left_curverad, car_offset)
        self.right_line.set_output_params(right_curverad, car_offset)
        output = self.draw_lines(warped, original, ploty)
        cv2.putText(output, "Car offset: %.2f m" % self.left_line.mean_car_offset, \
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(output, "Left curvature: %.2f m" % self.left_line.mean_curvature, \
            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(output, "Right curvature: %.2f m" % self.right_line.mean_curvature, \
            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        if debug:
            return output, sw_output
        return output

    def process_image_ex(self, warped, original, combined):
        """
            Process the image and generates additional data output.
        """
        output, sw_output = self.find_lines(warped, original, True)
        comb = cv2.bitwise_and(original, original, mask=combined)
        output1 = cv2.resize(comb, (640, 360), interpolation=cv2.INTER_AREA)
        output2 = cv2.resize(sw_output, (640, 360), interpolation=cv2.INTER_AREA)
        vis = np.zeros((720, 1280 + 640, 3))
        vis[:720, :1280, :] = output
        vis[:360, 1280:1920, :] = output1
        vis[360:720, 1280:1920, :] = output2
        return vis

class Line(object):
    """
        Line holder object.
    """

    def __init__(self):
        """
            Returns a line object.
        """
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.recent_fit = []
        self.best_fit = None
        self.current_fit = np.array([0, 0, 0], dtype='float')
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.allx = None
        self.ally = None
        self.last_fit_suspitious = False
        self.recent_curvatures = []
        self.mean_curvature = None
        self.recent_car_offsets = []
        self.mean_car_offset = None
        self.output_frames = 0

    def was_detected(self, next_x, next_curvature, next_fit, next_other_curvature, \
        next_other_fit, other_line_not_detected=False):
        """
            Evaluates if the line has been detected.
        """
        prev_detected = self.detected
        this_detected = self.best_fit is None or \
            ((np.abs(self.radius_of_curvature - next_curvature) < 5000 \
            or (self.radius_of_curvature > 5000 and next_curvature > 5000)) and \
            (np.abs(self.current_fit - next_fit) < [0.005, 2.0, 150.0]).all() and \
            (np.abs(next_other_curvature - next_curvature) < 5000  or \
            (next_other_curvature > 5000 and next_curvature > 5000)) and \
            (np.abs(next_other_fit[0] - next_fit[0]) < 0.001) and \
            (np.abs(next_other_fit[1] - next_fit[1]) < 0.5))
        self.detected = not prev_detected or not other_line_not_detected and this_detected

        if self.detected:
            if len(self.recent_xfitted) >= 4:
                self.recent_xfitted.pop(0)
            self.recent_xfitted.append(next_x)
            self.bestx = np.mean(self.recent_xfitted, axis=0)
            if len(self.recent_fit) >= 4 and not self.last_fit_suspitious:
                self.recent_fit.pop(0)
            if self.last_fit_suspitious:
                self.recent_fit.pop()
            self.last_fit_suspitious = not this_detected
            self.recent_fit.append(next_fit)
            self.best_fit = np.mean(self.recent_fit, axis=0)
            self.current_fit = next_fit
            self.radius_of_curvature = next_curvature

    def set_output_params(self, curvature, car_offset):
        """
            Sets the output paramters.
        """
        if self.detected:
            if len(self.recent_curvatures) >= 4 and not self.last_fit_suspitious:
                self.recent_curvatures.pop(0)
                self.recent_car_offsets.pop(0)
            if self.last_fit_suspitious:
                self.recent_car_offsets.pop()
                self.recent_curvatures.pop()
            self.recent_curvatures.append(curvature)
            self.recent_car_offsets.append(car_offset)
            if self.output_frames == 0:
                self.mean_curvature = np.mean(self.recent_curvatures)
                self.mean_car_offset = np.mean(self.recent_car_offsets)
            if self.output_frames >= 4:
                self.output_frames = 0
            else:
                self.output_frames += 1
