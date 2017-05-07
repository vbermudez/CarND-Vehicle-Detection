"""
    Utility functions module
"""
import cv2
import numpy as np
from sklearn.externals import joblib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import os

def read_image(path):
    """
        Reads an image from file.
    """
    return mpimg.imread(path)

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

def write_two_img(imgs, titles, name, output='./output_images', cmap1=None, cmap2=None):
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
    if cmap1 is not None:
        ax1.imshow(imgs[0], cmap=cmap1)
    elif is_grayscaled(imgs[0]):
        ax1.imshow(imgs[0], cmap='gray')
    else:
        ax1.imshow(imgs[0])
    if cmap2 is not None:
        ax2.imshow(imgs[1], cmap=cmap2)
    elif is_grayscaled(imgs[1]):
        ax2.imshow(imgs[1], cmap='gray')
    else:
        ax2.imshow(imgs[1])
    ax1.set_title(titles[0], fontsize=50)
    ax2.set_title(titles[1], fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(os.path.join(output, name), bbox_inches='tight')
    plt.close(f)

def convert_color(img, conv='RGB2YCrCb'):
    """
        Converts to a color space.
    """
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
        Computes the histogram of the color channels separately.
    """
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
        Draws boxes on the image.
    """
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy

def add_heat(heatmap, bbox_list):
    """
        Adds "heat" to a list of bounding boxes.
    """
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    """
        Applies a threshold on a heatmap.
    """
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    """
        Draws labeled boxes over detected cars.
    """
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img

def save_model(svc, file='./model.pkl'):
    """
        Saves the model.
    """
    joblib.dump(svc, file)
    print("Model saved.")

def load_model(file='./model.pkl'):
    """
        Loads the model.
    """
    print("Loading model...")
    svc = joblib.load(file)
    print("Model loaded.")
    return svc

def save_features(cars_feats, notcars_feats, file='./features.pkl'):
    """
        Saves features to disk.
    """
    feats = {"cars": cars_feats, "notcars": notcars_feats}
    joblib.dump(feats, file)
    print("Features saved.")

def load_features(file='./features.pkl'):
    """
        Load features from disk.
    """
    print("Loading features...")
    feats = joblib.load(file)
    print("Features loaded.")
    return feats["cars"], feats["notcars"]

def save_scaler(X_scaler, file='./xscaler.pkl'):
    """
        Saves X scaler to disk.
    """
    joblib.dump(X_scaler, file)
    print("X scaler saved.")

def load_scaler(file='./xscaler.pkl'):
    """
        Load X scaler from disk.
    """
    print("Loading X scaler...")
    X_scaler = joblib.load(file)
    print("X scaler loaded.")
    return X_scaler

def load_images():
    """
        Loads the cars and not-cars images.
    """
    cars = glob.glob("./vehicles/*/*.png")
    notcars = glob.glob("./non-vehicles/*/*.png")
    print("Car samples: {}.".format(len(cars)))
    print("Not car samples: {}.".format(len(notcars)))
    return cars, notcars



