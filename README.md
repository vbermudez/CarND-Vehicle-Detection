**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car1]: ./output_images/car0066.png
[car2]: ./output_images/car0084.png
[car3]: ./output_images/car0162.png
[notcar1]: ./output_images/notcar11.png
[notcar2]: ./output_images/notcar116.png
[notcar3]: ./output_images/notcar172.png
[carhogch1]: ./output_images/hog1_car0084.png
[carhogch2]: ./output_images/hog2_car0084.png
[carhogch3]: ./output_images/hog3_car0084.png
[notcarhogch1]: ./output_images/hog1_notcar116.png
[notcarhogch2]: ./output_images/hog2_notcar116.png
[notcarhogch3]: ./output_images/hog3_notcar116.png
[carspatialbin]: ./output_images/spatial_car0162.png
[notcarspatialbin]: ./output_images/spatial_notcar172.png
[carhistogram]: ./output_images/histogram_car0162.png
[notcarhistogram]: ./output_images/histogram_notcar172.png
[sliding1]: ./output_images/sliding_win_1test4.jpg
[sliding2]: ./output_images/sliding_win_1.5test1.jpg
[sliding3]: ./output_images/sliding_win_2.0test6.jpg
[sample1]: ./output_images/detections__test6.jpg
[sample2]: ./output_images/heatmap_test6.jpg
[sample3]: ./output_images/contour_test6.jpg
[heatmap1]: ./output_images/heatmap_1494781901.0230606.jpg
[heatmap2]: ./output_images/heatmap_1494781924.9642365.jpg
[heatmap3]: ./output_images/heatmap_1494781949.0266788.jpg
[heatmap4]: ./output_images/heatmap_1494781973.1248085.jpg
[heatmap5]: ./output_images/heatmap_1494782001.446343.jpg
[heatmap6]: ./output_images/heatmap_1494782026.051958.jpg
[accheatmap]: ./output_images/acc_heatmap_1494782026.051958.jpg
[bbox]: ./output_images/bbox_1494782026.051958.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 18 through 36 of the file [features.py](./carndlib/features.py)).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car1]|![car2]|![car3]
|------|-------|-------|
![notcar1]|![notcar2]|![notcar3]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Car:
![carhogch1]|![carhogch2]|![carhogch3]
|-----------|------------|------------|

Not car:
![notcarhogch1]|![notcarhogch2]|![notcarhogch3]
|--------------|---------------|---------------|

![carhistogram]|![carspatialbin]
|--------------|----------------|
![notcarhistogram]|![notcarspatialbin]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and tested using the last lesson's functions and trying to detect the cars. I've chosen the ones that, in my opinion, had better results in the overall process.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG, spatially binned color and historgram of colors as features. First, I've extracted all the features of each image, car and non car ones (lines 108 through 132 in [train.py](./carndlib/train.py) and 57 through 86 in [features.py](./carndlib/features.py)). To do so, I've transformed the images to the `YCrCb` color space (line 67 in [features.py](./carndlib/features.py), implemented within lines 219 through 235 in [utils.py](./carndlib/utils.py)). Then, I've extracted the mentioned features (lines 68 through 85 in [features.py](./carndlib/features.py)). 
Once extracted all the features, I've scaled them using a `StandardScaler` from `sklearn` library. 

In order to build both, the training and the test datasets, I've vertically stacked both features arrays into a single one; then I created a horizontally stacked array with ones for `cars` and zeros for `not cars`, as labels. I've used `train_test_split` from `sklearn` to randomize and split the data into training and test sets.

I've fitted the `LinearSVC` using the training set an tested it with the test set, obtaining an accuracy of 0.998873.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I've decided to search a defined area of the image, between 400 and 656 of Y axis, and using three different scale (1., 1.5 and 2.). Here some examples:

![sliding1]|![sliding2]|![sliding3]
|----------|-----------|-----------|

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![sample1]|![sample2]|![sample3]
|---------|----------|----------|
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output.mp4) and [another one](./output_2.mp4) with combined lane detection and car finding and tracking projects.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in last 10 frames of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![heatmap1]|![heatmap2]|![heatmap3]
|----------|-----------|-----------|
![heatmap4]|![heatmap5]|![heatmap6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![accheatmap]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![bbox]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There is a lot to talk about this project. First about it: how incredibly is the amount of time it takes to process the video. I assume that it's because my own inexpertise in this field. But, I think it cannot be much more quick than my approach. I think that the work can be parallelized, but again, it won't raise the performance by much. It is not usable in real time.
In the other hand, it looks pretty efficient on detecting the vehicles. Anyway, the result is highly enjoyable!

