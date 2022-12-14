import os
import glob

folders = ["data/masks_needle3/", "data/imgs_needle3/"]

for folder in folders:
    fn_list = glob.glob(folder + "*.png")
    for fn in fn_list:
        tmp = fn.split(".png")
        fn_new = tmp[0] + tmp[1] + ".png"
        os.rename(fn, fn_new)
