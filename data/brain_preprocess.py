import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from PIL import Image

path = "kaggle_3m/"
imgs_path = "imgs_brain/"
masks_path = "masks_brain/"

file_list = glob.glob(path + "**/*_mask.tif")


for i, filename in enumerate(file_list):

    # find mask
    mask = cv2.imread(filename)

    # remove no tumor mask
    if np.max(mask) == 0:
        continue

    # format and save
    mask = np.sum(mask, axis=2)
    mask = mask.astype(bool)
    filename = filename.split("/")[-1].split(".")[0]
    mask_filename = masks_path + filename + ".png"
    im = Image.fromarray(mask)
    im.save(mask_filename)
    # print(f"save image #{i}  {mask_filename}")

    # find img
    img_filename = filename.split("_mask")[0]
    img_filename = glob.glob(path + "**/" + img_filename + ".tif")[0]
    img = cv2.imread(img_filename)

    # format and save
    img_filename = img_filename.split("/")[-1].split(".")[0]
    img_filename = imgs_path + img_filename + ".png"
    im = Image.fromarray(img)
    im.save(img_filename)
    print(f"save image #{i}  {img_filename}")
