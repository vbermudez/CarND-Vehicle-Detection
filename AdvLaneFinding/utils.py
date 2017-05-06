"""
    Utility functions module
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
import numpy as np

def grayscale(img):
    """
        Returns an image grayscaled.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def thresh(img, thresh_min, thresh_max):
    """
         Applies the specified threshold to an image.
    """
    ret = np.zeros_like(img)
    ret[(img >= thresh_min) & (img <= thresh_max)] = 1
    return ret

def rgb2hls(img):
    """
        Converts to HLS color space.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def rgb2hsv(img):
    """
        Converts to HSV color space.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def rgb2lab(img):
    """
        Converts to LAB color space.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

def rgb2luv(img):
    """
        Converts to LUV color space.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def select_channel(img, space='hls', channel='s'):
    """
        Select a channel from a image.
    """
    ich = space.index(channel)
    return img[:, :, ich]

def extract_white_yellow(img):
    """
        Extracts white and yellow pixels.
    """
    hsv = rgb2hsv(img)
    b = np.zeros((img.shape[0], img.shape[1]))
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    t_yellow_H = thresh(H, 10, 30)
    t_yellow_S = thresh(S, 50, 255)
    t_yellow_V = thresh(V, 150, 255)
    t_white_R = thresh(R, 225, 255)
    t_white_V = thresh(V, 230, 255)
    b[(t_yellow_H == 1) & (t_yellow_S == 1) & (t_yellow_V == 1)] = 1
    b[(t_white_R == 1) | (t_white_V == 1)] = 1
    return b

def equalize(img):
    """
        Equalizes the histogram of an image.
    """
    return cv2.equalizeHist(img[:, :, 0])

def gaussian_blur(img, kernel_size=5):
    """
        Applies a Gaussian Noise kernel
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def get_region(image):
    """
        Returns the part which is likely to contain lane lines.
    """
    bottom_left = (image.shape[1] * .15, image.shape[0])
    bottom_right = (image.shape[1] * .93, image.shape[0])
    top_left = (image.shape[1] * .4, image.shape[0] / 2)
    top_right = (image.shape[1] * .6, image.shape[0] / 2)
    white = np.zeros_like(image)
    points = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(white, points, 255)
    return cv2.bitwise_and(image, white)

def remove_atypical(x, y, perc=5, horitzontal=True):
    """
        Removes atypical values based on percentile.
    """
    if len(x) == 0 or len(y) == 0:
        return x, y

    x = np.array(x)
    y = np.array(y)

    if horitzontal:
        lower_bound = np.percentile(x, perc)
        upper_bound = np.percentile(x, 100 - perc)
        selection = (x >= lower_bound) & (x <= upper_bound)
    else:
        lower_bound = np.percentile(y, perc)
        upper_bound = np.percentile(y, 100 - perc)
        selection = (y >= lower_bound) & (y <= upper_bound)
    return x[selection], y[selection]

def write_image(img, path):
    """
        Writes an image into a file.
    """
    cv2.imwrite(path, img)

def is_grayscaled(img):
    """
        Returns True if an images is grayscaled
    """
    return not (len(img.shape) == 3 and img.shape[2] == 3)

def plot_points(img, one, two, three, four, name, output='./output_images'):
    """
        Writes a image with four points.
    """
    if is_grayscaled(img):
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.plot(one[0], one[1], 'r.')
    plt.plot(two[0], two[1], 'g.')
    plt.plot(three[0], three[1], 'b.')
    plt.plot(four[0], four[1], 'y.')
    plt.savefig(os.path.join(output, name), bbox_inches='tight')
    plt.close('all')

def plot_histogram(img, histogram, name, output='./output_images'):
    """
        Plots the histogram over the image.
    """
    f, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    f.tight_layout()
    if is_grayscaled(img):
        ax1.imshow(img, cmap='gray')
    else:
        ax1.imshow(img)
    ax2.plot(histogram)
    ax2.set_title('Histogram', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(os.path.join(output, name), bbox_inches='tight')
    plt.close(f)

def write_two_img(imgs, titles, name, output='./output_images'):
    """
        Writes two images in a single plot.

        Attributes:
            imgs: Array of 2 images, one for each subplot.
            titles: Array of 2 titles, one for each subplot.
            name: Name of the file
            output: Output directory
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    if is_grayscaled(imgs[0]):
        ax1.imshow(imgs[0], cmap='gray')
    else:
        ax1.imshow(imgs[0])
    if is_grayscaled(imgs[1]):
        ax2.imshow(imgs[1], cmap='gray')
    else:
        ax2.imshow(imgs[1])
    ax1.set_title(titles[0], fontsize=50)
    ax2.set_title(titles[1], fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(os.path.join(output, name), bbox_inches='tight')
    plt.close(f)

def read_image(path):
    """
        Reads an image from file.
    """
    return mpimg.imread(path)

def list_dir(path):
    """
        Lists all the files in a directory.
    """
    return glob.glob(os.path.join(path, '*'))
