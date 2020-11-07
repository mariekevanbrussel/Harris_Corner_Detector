#%%
import cv2
import numpy as np
from scipy.signal import convolve2d
from harris_corner_detector import harris_corner_detector

def harris_corner_detector_plot(img_color_orig, thresh, sigma=1, window_size=5):

    img_color = img_color_orig.copy()
    Points = harris_corner_detector(img_color, thresh, sigma, window_size)
    radius = 1
    color = (0, 255, 0)  # Green
    thickness = 1

    for p in Points:
        cv2.circle(img_color, (p[1], p[0]), radius, color, thickness)

    return img_color



