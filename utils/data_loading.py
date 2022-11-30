import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(
            images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(
                f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, pil_mask, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        if not pil_mask == None:
            transform = A.Compose([
                A.Flip(p=0.2),
                A.CropNonEmptyMaskIfExists(height=700, width=700, p=.2),
                A.GridDistortion(p=0.2),
                A.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2,
                              hue=0.05, always_apply=False, p=0.2),

                A.Resize(newH, newW, interpolation=2)
            ])
        else:
            transform = A.Resize(newH, newW, interpolation=2)

        img_ndarray = np.asarray(pil_img)

        if not pil_mask == None:
            mask_ndarray = np.asarray(pil_mask)
            transformed = transform(
                image=img_ndarray, mask=mask_ndarray.astype(np.uint8))
            img_ndarray = transformed['image'] / 255
            mask_ndarray = transformed['mask'].astype(bool)

        else:
            transformed = transform(image=img_ndarray)
            img_ndarray = transformed['image'] / 255

        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not pil_mask == None:
            return (img_ndarray, mask_ndarray)
        else:
            return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(
            img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(
            mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img, mask = self.preprocess(img, mask, self.scale)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
