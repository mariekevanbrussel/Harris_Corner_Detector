#%%
import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

def harris_corner_detector(img_color, thresh, sigma=1, window_size=5):

  img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
  gx = np.array([1, 0, -1]).reshape(1, 3)
  gy = gx.T

  #%% Derivative
  Ix_2 = convolve2d(img, gx, mode='same')
  Iy_2 = convolve2d(img, gy, mode='same')

  #%% Squared & Product
  Ix_sq = Ix_2**2
  Iy_sq = Iy_2**2
  I_prod = Ix_2 * Iy_2

  height, width = Ix_sq.shape

  A = gaussian_filter(Ix_sq, sigma, order=0)
  C = gaussian_filter(Iy_sq, sigma, order=0)
  B = gaussian_filter(I_prod, sigma, order=0)

  #Calculate H and determine the corner points
  H = np.zeros((height, width), np.float32)
  Points = []

  for row in range(height):
    for col in range(width):
      H[row][col] = (A[row][col] * C[row][col] -
                     B[row][col]**2) - 0.04 * (A[row][col] + C[row][col])**2
      value = H[row][col]
      if value > thresh:
        row_l = int(max(0, row - (window_size - 1) / 2))
        row_r = int(min(height, row + (window_size - 1) / 2 + 1))
        col_l = int(max(0, col - (window_size - 1) / 2))
        col_r = int(min(width, col + (window_size - 1) / 2 + 1))
        if (value == np.max(H[row_l:row_r, col_l:col_r])) & (np.count_nonzero(
            H[row_l:row_r, col_l:col_r] == value) == 1):
          Points.append((row, col))

  return Points
