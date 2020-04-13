# SDE-Project_1-Finding_Lane_on_the_road
This project is develop a lane line detector using python and OpenCV. 

The inputs are video stream and outputs are shown in gif. 

### White Lane
![](https://github.com/garychian/SDE-Project_1-Finding_Lane_on_the_road/blob/master/White.gif)

### Yellow Lane
![](https://github.com/garychian/SDE-Project_1-Finding_Lane_on_the_road/blob/master/yellow.gif)

### Challenge(curve lane)
![](https://github.com/garychian/SDE-Project_1-Finding_Lane_on_the_road/blob/master/Challenge.gif)

## Algorithm Pipeline
1. Import Dependencies

- Python 3
- Numpy
- OpenCV
- Matplotlib
- MoviePy

2. `grayscale(img)`

Convert the image to grayscale

3. `Canny(img, low_threshold, high_threshold)` transform

Before running `canny`, Guassian smoothing is essentially way of supperssing noise and spurious gradients by averaging
Note: `cv2Canny()`actually applies Guassian smooting internally. `kernel_size` for Guassian smooting to be any odd number, A Large `kernel_size`implies averaging,or smoothing, over a **larger** area. Normally, `kernel_size = 3`

4. `region_of_interest(img, vertices)`

Which applies an image mask, Only keeps the region of the image defined by the polygon
formed from `vertices`. The rest of the image is set to black.

