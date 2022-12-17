import os

#numpy
import numpy as np
# cv
import cv2 as cv

def connected_component_filtering( bin_img, N_keep: int = 1 ):
    """ Keep the largest 'N_keep' connected components

        @param bin_img: binary image for connected component analysis
        @param N_keep: the number of components to keep

        @return: segmented out connected components
    """

    # run connected components (0 is bg)
    num_labels, labels = cv.connectedComponents( bin_img )

    # determine number of instances per label
    num_instances_labels = [ np.count_nonzero( labels == lbl ) for lbl in range( num_labels ) ]

    # N largest label that is not the background
    labels_sorted = np.argsort( num_instances_labels[ 1: ] ) + 1
    labels_Nkeep = labels_sorted[ :-N_keep - 1:-1 ]

    bin_img_cc = np.isin( labels, labels_Nkeep )

    return bin_img_cc



path_masks = "all_masks_last/"

for filename in os.listdir(path_masks):
    mask = cv.imread(os.path.join(path_masks,filename),-1)
    print(np.unique(mask))
    cc_mask = connected_component_filtering(mask, 1)
    print(np.unique(cc_mask))
    
    cv.imwrite("all_masks_last_filterted/"+filename,255*cc_mask.astype(int))

