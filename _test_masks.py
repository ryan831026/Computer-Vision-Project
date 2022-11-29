import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from os import listdir
from os.path import splitext
import numpy as np



### Needle images 24 bit to 8 bit binary format image
dir_mask = './data/masks_needle_3ch/'
targetdir_mask = './data/masks_needle/'
path_mask = Path(dir_mask)

ids = [splitext(file)[0] for file in listdir(dir_mask) if not file.startswith('.')]


list_imgs=listdir(dir_mask)
for im in list_imgs:
    img=Image.open( dir_mask+im ).convert('1')#.point(lambda p: 1 if p==225 else 0)
    # pixels = img.load()
    # for i in range(img.size[0]): # for every pixel:
    #     for j in range(img.size[1]):
    #         if pixels[i,j] == (255):
    #             pixels[i,j] = (1)

    img.save(targetdir_mask+im)
    # print(np.where(np.array(img) ==1)[0].shape)
    print(img.info)
    print(img.mode)



