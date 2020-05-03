# SDE-Project_1-Finding_Lane_on_the_road

## Overview
Update 5/1/2020: Base the previous project, this is the optimized algorithm for lane_finding project. This method make the algorithm robust and stable which can handle more cases. 

### Pipeline
* Calibration Camera
* Threshold Binary
* Color Transform 
* Perspective Transform
* Detect lane pixels using Histogram 
* Measuring Curvature

### Model details



update: 4/2/2020
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

3. `Canny(img, low_threshold, high_threshold)` Detect Edges

Before running `canny`, Guassian smoothing is essentially way of supperssing noise and spurious gradients by averaging
Note: `cv2Canny()`actually applies Guassian smooting internally. `kernel_size` for Guassian smooting to be any odd number, A Large `kernel_size`implies averaging,or smoothing, over a **larger** area. Normally, `kernel_size = 3`

4. `region_of_interest(img, vertices)` Mask edges to area of interest

Which applies an image mask, Only keeps the region of the image defined by the polygon
formed from `vertices`. The rest of the image is set to black.

5. `hough_lines(image_Mask, rho, theta, threshold, min_line_len, max_line_gap)`Detect Hough lines and draw lines on blank image. 

6. `weighted_img(img, initial_img, α=0.8, β=1., λ=0.)` Draw lines on original image

	`img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!

## Chanllenge 

When I apply the algorithm to the `challenge.mp4` I found the results are not shown very well. The algorithm cannot hand the curve yellowleft side lane. So an optimized algorithm are created. 

For people who want to run algorithm: go to cmd and try code down below

`python Project1_Challenge_parts_1.py -i challenge.mp4 -o extra.mp4`

### What changed to improved the robustness?

1. Add `filter_colors()` (before grayscale) to filter the image to include only yellow and white pixels

2. Use trapezoid-shaped `region-of-interest`

```python
# previous code
vertices = np.array([[(0,imshape[0]),(450, 320), (490, 320), (imshape[1],imshape[0])]], dtype=np.int32)

# optimized code
vertices = np.array([[\
		((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),\
		((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
		(imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
		(imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]]\
		, dtype=np.int32)
```

3. Optimized the `draw_line()` function. 

	1. Only care about lines where abs(slope) > slope_threshold(0.5)

	2. Split lines into right_lines and left_lines, representing the right and left lane 	lines

	3. Right/left lane lines must have positive/negative slope, and be on the right/left 	half of the image

	4. Run linear regression to find best fit line for right and left lane lines

	5. Find 2 end points for right and left lines, used for drawing the line
		y = m*x + b --> x = (y - b)/m


## Files in the folder 

`Project 1_Finding_Lane_lines_on_the_road.ipynb` is the most of code, it processed images `white.mp4` and `yellow.mp4`

`Project1_Challenge_parts_1.py` is optimized for `challenge.mp4`