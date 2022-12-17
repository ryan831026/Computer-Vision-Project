import os

#numpy
import numpy as np
# cv
import cv2
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError


import matplotlib.pyplot as plt

#needle_reconstruction
from needle_reconstruction import StereoRefInsertionExperiment

class NeedleSegmenter:
    def __init__(self):
        stereo_reconstruction_param_file = "reconstruction_params.json"
        # get Stereo Needle Shape Reconstruction object
		
        
        try:
            self.needle_reconstructor = StereoRefInsertionExperiment.load_json( stereo_reconstruction_param_file )
            print( "Successfully loaded Stereo Needle Shape Reconstructor: \n" + str( self.needle_reconstructor))
        # try
        except Exception as e:
            print( f"{stereo_reconstruction_param_file} is not a valid stereo reconstruction parameter file." )
            raise e
		# except



        
        ### FOR STEREO IMAGE
        path_left_imgs = "../left_imgs/"
        path_right_imgs = "../right_imgs/"
        path_ref_left_imgs = "../ref_left_imgs/"
        path_ref_right_imgs = "../ref_right_imgs/"

        for i,filename in enumerate(os.listdir(path_left_imgs)):
            ### Load ref images
            ref_left_filename = filename[:19]+'.png' 
            ref_right_filename = filename[:19] +'.png'
            ref_left = cv2.imread(os.path.join(path_ref_left_imgs,ref_left_filename),1)
            ref_right = cv2.imread(os.path.join(path_ref_right_imgs,ref_right_filename),1)
            self.needle_reconstructor.needle_reconstructor.load_image_pair(ref_left,ref_right, reference=True)
            

            ### Load target images
            left_filename = filename 
            right_filename = filename
            left_img = cv2.imread(os.path.join(path_left_imgs,left_filename),1)
            right_img = cv2.imread(os.path.join(path_right_imgs,right_filename),1)
            self.needle_reconstructor.needle_reconstructor.load_image_pair(left_img, right_img, left_filename, right_filename, reference=False)
        

        
        
            try:
                
                imgs = self.needle_reconstructor.needle_reconstructor.reconstruct_needle()
                # cv2.imwrite('check_sub.png',imgs['sub'])
                # cv2.imwrite('check_seg-init.png',imgs['seg-init'])
                # cv2.imwrite('check_seg.png',imgs['seg'])
                # cv2.imwrite('check_roibo.png',imgs['roibo'])
                # cv2.imwrite('check_skel.png',imgs['skel'])
                # cv2.imwrite('check_contours.png',imgs['contours'])
                # cv2.imwrite('check_contours-match.png',imgs['contours-match'])
                
            except Exception as e:
                print("EXCEPTION")
            



def main():
    needle_segmenter = NeedleSegmenter()



if __name__ == "__main__":
    main()

# if __main__