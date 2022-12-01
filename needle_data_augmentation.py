import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import albumentations as A

path_needle = 'data/imgs_needle2/'
path_mask = 'data/masks_needle2/'

# for i, filename_full in enumerate(glob.glob(os.path.join(path_needle, '*.png'))):
#     filename = filename_full.split('/')[-1]
#     print(filename)

filename_full = glob.glob(os.path.join(path_needle, '*.png'))[-1]
filename = filename_full.split('/')[-1]
img = cv2.imread(filename_full)

fsplit = filename.split('.')
print(path_mask + fsplit[0] + '.' + fsplit[1] + '_mask.png')
filename_full = path_mask + fsplit[0] + '.' + fsplit[1] + '_mask.png'
mask = cv2.imread(filename_full)


plt.figure(1)
plt.subplot(221)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(222)
plt.imshow(mask, cmap='gray')

print(img.dtype)

# Declare an augmentation pipeline
transform = A.Compose([
    A.Flip(p=1),
    A.CropNonEmptyMaskIfExists(height=700, width=700, p=1),
    A.GridDistortion(p=1),
    A.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1,
                  hue=0.1, always_apply=False, p=1),

    A.Resize(572, 572, interpolation=2)
])

# transform = A.Resize(200, 200, interpolation=2)

# Read an image with OpenCV and convert it to the RGB colorspace

# Augment an image
transformed = transform(image=img, mask=mask)

plt.subplot(223)
plt.imshow(cv2.cvtColor(transformed["image"], cv2.COLOR_BGR2RGB))
plt.subplot(224)
plt.imshow(cv2.cvtColor(transformed["mask"], cv2.COLOR_BGR2RGB))
plt.show()
