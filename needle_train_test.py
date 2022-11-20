import cv2
import numpy as np

# img = cv2.imread('data/masks_needle/left_0.02_mask.png')
img = cv2.imread('data/masks/0cdf5b5d0ce1_01_mask.gif')
# img = cv2.imread('data/imgs/0cdf5b5d0ce1_01.jpg')
print(type(img))
print(img.shape)
print(np.max(img))
print(np.min(img))
print(np.unique(img))