import argparse
import logging
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

from predict import predict_img, mask_to_image


def get_args():
    parser = argparse.ArgumentParser(
        description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT',
                        nargs='+', help='Folder of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT',
                        nargs='+', help='Folder of output images', required=True)
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true',
                        default=False, help='Use bilinear upsampling')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    in_folder = args.input[0]
    out_folder = args.output[0]

    file_list = glob.glob(os.path.join(in_folder, '*.png'))

    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(file_list):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # plot
        img = cv2.imread(filename)
        img_needle = np.zeros(img.shape, dtype=int)
        img_needle[:, :, 2] = mask[1, :, :] * 255
        overlayed_img = img + 0.5 * img_needle
        overlayed_img[overlayed_img > 255] = 255
        overlayed_img = overlayed_img.astype(np.uint8)
        try:
            os.mkdir(out_folder)
        except:
            pass
        mask_filename_full = out_folder + filename.split('/')[2]
        cv2.imwrite(mask_filename_full, overlayed_img)
        print(f'save image #{i+1}  {mask_filename_full}')
