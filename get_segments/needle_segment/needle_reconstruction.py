"""
Created on Aug 26, 2021

This is a library/script to perform stereo needle reconstruciton and process datasets


@author: Dimitri Lezcano

"""
import argparse
import json
import glob
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any, Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import stereo_needle_proc as stereo_needle


@dataclass
class ImageROI:
    image: np.ndarray
    roi: List[ int ] = field( default_factory=list )
    blackout: List[ int ] = field( default_factory=list )


# dataclass: ImageROI

@dataclass
class StereoPair:
    left: Any = None
    right: Any = None

    def __eq__( self, other ):
        if isinstance( other, StereoPair ):
            retval = (self.left == other.left, self.right == other.right)

        elif isinstance( other, (tuple, list) ):
            retval = (self.left == other[ 0 ], self.right == other[ 1 ])

        else:
            retval = (self.left == other, self.right == other)

        return retval

    # __eq__

    @staticmethod
    def from_image( left: str = None, right: str = None ):
        """ Generate a StereoPair object from to image files"""
        sp = StereoPair()
        sp.load_image_pair( left=left, right=right )

        return sp

    # from_image

    def load_image_pair( self, left: str = None, right: str = None ):
        """ Load image pair of image files"""
        if left is not None:
            self.left = cv.imread( left, cv.IMREAD_COLOR )

        if right is not None:
            self.right = cv.imread( right, cv.IMREAD_COLOR )

    # load_image_pair

    def set( self, left, right ):
        """ Function to set the stereo pair"""
        self.left = left
        self.right = right

    # set_pair


# dataclass: StereoPair

@dataclass
class StereoImagePair:
    left: ImageROI
    right: ImageROI


# dataclass: StereoImagePair

class StereoRefInsertionExperiment:
    directory_pattern = r".*[/,\\]Insertion([0-9]+)[/,\\]([0-9]+).*"  # data directory pattern

    def __init__( self, stereo_param_file: str, insertion_depths: list = None,
                  insertion_numbers: list = None, roi: tuple = None, blackout: tuple = None, contrast: tuple = None,
                  window_size: np.ndarray = None,  alpha: float = None, zoom: float = None, sub_thresh: float = None):
        stereo_params = stereo_needle.load_stereoparams_matlab( stereo_param_file )
        
        # self.data_directory = os.path.normpath( data_dir )
        # self.insertion_numbers = insertion_numbers
        
        # if insertion_depths is None:
        #     self.insertion_depths = None

        # else:  # make sure non-negative insertion depth
        #     self.insertion_depths = [ 0 ] + list( filter( lambda d: d > 0, insertion_depths ) )

        self.needle_reconstructor = StereoNeedleRefReconstruction( stereo_params, None, None, None, None )

        # set the datasets ROI
        if roi is not None:
            self.needle_reconstructor.roi.left = roi[ 0 ]
            self.needle_reconstructor.roi.right = roi[ 1 ]

        # if

        # set the image blackout regions
        if blackout is not None:
            self.needle_reconstructor.blackout.left = blackout[ 0 ]
            self.needle_reconstructor.blackout.right = blackout[ 1 ]

        # if

        # set the image contrast enhancements
        if contrast is not None:
            self.needle_reconstructor.contrast.left = contrast[ 0 ]
            self.needle_reconstructor.contrast.right = contrast[ 1 ]

        # if

        if window_size is not None:
            self.needle_reconstructor.window_size = window_size

        # if

        if alpha is not None:
            self.needle_reconstructor.alpha = alpha

        # if

        if zoom is not None:
            self.needle_reconstructor.zoom = zoom
            
        # if

        if sub_thresh is not None:
            self.needle_reconstructor.sub_thresh = sub_thresh
            
        # if



        # configure the dataset
        # self.dataset, self.processed_data = self.configure_dataset( self.data_directory, self.insertion_depths,
        #                                                             self.insertion_numbers )

    # __init__

    @property
    def processed_images( self ):
        return self.needle_reconstructor.processed_images

    # processed_images

    @property
    def processed_figures( self ):
        return self.needle_reconstructor.processed_figures

    # processed_figures

    @property
    def stereo_params( self ):
        return self.needle_reconstructor.stereo_params

    # stereo_params

    @classmethod
    def configure_dataset( cls, directory: str, insertion_depths: list, insertion_numbers: list ) -> (list, list):
        """
            Configure a dataset based on the directory:

            :param directory: string of the main data directory
            :param insertion_depths: a list of insertion depths that are to be processed.
            :param insertion_numbers: a list of insertion numbers to process

        """
        dataset = [ ]
        processed_dataset = [ ]

        if directory is None:
            return dataset

        # if

        directories = glob.glob( os.path.join( directory, 'Insertion*/*/' ) )

        # iterate over the potential directories
        for d in directories:
            res = re.search( cls.directory_pattern, d )

            if res is not None:
                insertion_num, insertion_depth = res.groups()
                insertion_num = int( insertion_num )
                insertion_depth = float( insertion_depth )

                # only include insertion depths that we want to process
                if (insertion_depths is None) and (insertion_numbers is None):  # take all data
                    dataset.append( (d, insertion_num, insertion_depth) )

                elif insertion_depths is None:  # take all depths
                    if insertion_num in insertion_numbers:
                        dataset.append( (d, insertion_num, insertion_depth) )

                elif insertion_numbers is None:
                    if insertion_depth in insertion_depths:  # take all insertion trials
                        dataset.append( (d, insertion_num, insertion_depth) )

                elif insertion_depth in insertion_depths and insertion_num in insertion_numbers:  # be selective
                    dataset.append( (d, insertion_num, insertion_depth) )

                if os.path.isfile( os.path.join( d, 'left-right_3d-pts.csv' ) ):
                    # load the processed data
                    pts_3d = np.loadtxt( os.path.join( d, 'left-right_3d-pts.csv' ), delimiter=',' )
                    processed_dataset.append( (d, insertion_num, insertion_depth, pts_3d) )  # processed_dataset

                # if

            # if
        # for

        return dataset, processed_dataset

    # configure_dataset

    @staticmethod
    def load_json( filename: str ):
        """ 
        This function is used to load a StereoRefInsertionExperiment class from a saved JSON file.
        
        Args:
            - filename: str, the input json file to be loaded.
            
        Returns:
            A StereoRefInsertionExperiment Class object with the loaded json parameters.
        
        """
        # load the data from the json file to a dict
        print(filename)
        with open( filename, 'r' ) as json_file:
            data = json.load( json_file )

        # with

        stereoParamFile = data.get( 'stereo parameters mat file location', None )

        if 'ROI' in data.keys():
            keysROI=data['ROI'].keys()
            if ('left' in keysROI) and ('right' in keysROI):
                leftROI = [data["ROI"]["left"][0:2],data["ROI"]["left"][2:4]]
                rightROI = [data["ROI"]["right"][0:2],data["ROI"]["right"][2:4]]

            # if

        # if
        else:
            leftROI = []
            rightROI = []

        # else

        if 'blackout' in data.keys():
            keysBlackout=data['blackout'].keys()
            if ('left' in keysBlackout) and ('right' in keysBlackout):
                leftBlackout = data["blackout"]["left"]
                rightBlackout = data["blackout"]["right"]

            # if

        # if
        else:
            leftBlackout = []
            rightBlackout = []

        # else

        

        # else

        if 'contrast enhance' in data.keys():
            keysContrast=data['contrast enhance'].keys()
            if ('left' in keysContrast) and ('right' in keysContrast):
                leftContrastEnhance = tuple(data["contrast enhance"]["left"])
                rightContrastEnhance = tuple(data["contrast enhance"]["right"])

            # if
            
        # if
        else:
            leftContrastEnhance = None
            rightContrastEnhance = None

        # else

        if 'window size' in data.keys():
            windowSize = tuple(data['window size'])

        # if

        else:
            windowSize = None

        # else

        zoom = data.get( 'zoom', None )
        alpha = data.get( 'alpha', None )
        subtractThr = data.get( 'subtract threshold', None )

        # instantiate the StereoRefInsertionExperiment class object
        needle_reconstructor = StereoRefInsertionExperiment( stereoParamFile, 
                                                                roi=(leftROI, rightROI), 
                                                                blackout=(leftBlackout,rightBlackout),
                                                                contrast=(leftContrastEnhance, rightContrastEnhance),
                                                                window_size=windowSize,
                                                                zoom=zoom,
                                                                alpha=alpha,
                                                                sub_thresh = subtractThr
                                                            )

        # return the instantiation
        return needle_reconstructor

    # load_json


# class: StereoRefInsertionExperiment


class StereoNeedleReconstruction( ABC ):
    """ Basic class for stereo needle reconstruction"""
    save_fbase = 'left-right_{:s}'

    def __init__( self, stereo_params: dict, img_left: np.ndarray = None, img_right: np.ndarray = None ):
        self.stereo_params = stereo_params
        self.image = StereoPair( img_left, img_right )

        self.roi = StereoPair( [ ], [ ] )
        self.blackout = StereoPair( [ ], [ ] )
        self.contrast = StereoPair( (1, 0), (1, 0) )  # (alpha, beta): alpha * image + beta

        self.needle_shape = None
        self.img_points = StereoPair( None, None )
        self.img_bspline = StereoPair( None, None )
        self.processed_images = { }
        self.processed_figures = { }

    # __init__

    @staticmethod
    def contrast_enhance( image: np.ndarray, alpha: float, beta: float ):
        """ Perform contrast enhancement of an image

            :param image: the input image
            :param alpha: the scaling term for contrast enhancement
            :param beta:  the offset term for contrast enhancement

            :returns: the contrast enhanced image as a float numpy array
        """
        return np.clip( alpha * (image.astype( float )) + beta, 0, 255 )

    # contrast_enhance

    def load_image_pair( self, left_img: np.ndarray = None, right_img: np.ndarray = None , left_filename: str = "", right_filename: str = "",):
        """ Load the image pair. If the one of the images is none, that image will not be loaded

            :param left_img: (Default = None) np.ndarray of the left image
            :param right_img: (Default = None) np.ndarray of the right image

        """

        if left_img is not None:
            self.image.left = left_img
            self.left_filename = left_filename
            

        # if

        if right_img is not None:
            self.image.right = right_img
            self.right_filename = right_filename

        # if

    # load_image_pair

    @abstractmethod
    def reconstruct_needle( self, **kwargs ) -> np.ndarray:
        """
            Reconstruct the 3D needle shape from the left and right image pair

        """
        pass

    # reconstruct_needle

    def save_3dpoints( self, outfile: str = None, directory: str = '', verbose: bool = False ):
        """ Save the 3D reconstruction to a file """

        if self.needle_shape is not None:
            if outfile is None:
                outfile = self.save_fbase.format( '3d-pts' ) + '.csv'

            # if

            outfile = os.path.join( directory, outfile )

            np.savetxt( outfile, self.needle_shape, delimiter=',' )
            if verbose:
                print( "Saved reconstructed shape:", outfile )

            # if

        # if

    # save_3dpoints

    def save_processed_images( self, directory: str = '.' ):
        """ Save the images that have now been processed

            :param directory: (Default = '.') string of the directory to save the processed images to.
        """
        # the format string for saving the figures
        save_fbase = os.path.join( directory, self.save_fbase )

        if self.processed_images is not None:
            for key, img in self.processed_images.items():
                cv.imwrite( save_fbase.format( key ) + '.png', img )
                print( "Saved figure:", save_fbase.format( key ) + '.png' )

            # for
        # if

        if self.processed_figures is not None:
            for key, fig in self.processed_figures.items():
                fig.savefig( save_fbase.format( key + '-fig' ) + '.png' )
                print( "Saved figure:", save_fbase.format( key + '-fig' ) + '.png' )

            # for
        # if

    # save_processed_images


# class: StereoNeedleReconstruction


class StereoNeedleRefReconstruction( StereoNeedleReconstruction ):
    """ Class for Needle Image Reference Reconstruction """

    def __init__( self, stereo_params: dict, img_left: np.ndarray = None, img_right: np.ndarray = None,
                  ref_left: np.ndarray = None, ref_right: np.ndarray = None ):
        super().__init__( stereo_params, img_left, img_right )
        self.reference = StereoPair( ref_left, ref_right )

    # __init__

    def load_image_pair( self, left_img: np.ndarray = None, right_img: np.ndarray = None, left_filename: str = "", right_filename: str = "",  reference: bool = False ):
        """ Load the image pair. If the one of the images is none, that image will not be loaded

            :param left_img: (Default = None) np.ndarray of the left image
            :param right_img: (Default = None) np.ndarray of the right image
            :param reference: (Default = False) whether we are loading the reference image or not
        """
        if not reference:
            super().load_image_pair( left_img, right_img ,left_filename, right_filename )

        # if
        else:
            if left_img is not None:
                self.reference.left = left_img

            # if

            if right_img is not None:
                self.reference.right = right_img

            # if

        # else

    # load_image_pair

    def reconstruct_needle( self ) -> np.ndarray:
        """
            Reconstruct the needle shape

            Keyword arguments:
                window size: 2-tuple of for window size of the stereo template matching (must be odd)
                zoom:        the zoom value for for the template maching algorithm
                alpha:       the alpha parameter in stereo rectification
                sub_thresh:  the threshold value for the reference image subtraction

        """
        # keyword argument parsing
        # window_size = kwargs.get( 'window_size', (201, 51) )
        # zoom = kwargs.get( 'zoom', 1.0 )
        # alpha = kwargs.get( 'alpha', 0.6 )
        # sub_thresh = kwargs.get( 'sub_thresh', 60 )
        # proc_show = kwargs.get( 'proc_show', False )

        window_size = self.window_size
        zoom = self.zoom
        alpha = self.alpha
        sub_thresh = self.sub_thresh

        # perform contrast enhancement
        ref_left = self.contrast_enhance( self.reference.left, self.contrast.left[ 0 ],
                                          self.contrast.left[ 0 ] ).astype( np.uint8 )
        img_left = self.contrast_enhance( self.image.left, self.contrast.left[ 0 ],
                                          self.contrast.left[ 1 ] ).astype( np.uint8 )
        ref_right = self.contrast_enhance( self.reference.right, self.contrast.right[ 0 ],
                                           self.contrast.left[ 0 ] ).astype( np.uint8 )
        img_right = self.contrast_enhance( self.image.right, self.contrast.right[ 0 ],
                                           self.contrast.right[ 1 ] ).astype( np.uint8 )

        # perform stereo reconstruction
        # pts_3d, pts_l, pts_r, bspline_l, bspline_r, imgs, figs = \
        imgs = \
            stereo_needle.needle_reconstruction_ref( img_left, ref_left, self.left_filename,
                                                     img_right, ref_right, self.right_filename,
                                                     stereo_params=self.stereo_params, recalc_stereo=True,
                                                     bor_l=self.blackout.left, bor_r=self.blackout.right,
                                                     roi_l=self.roi.left, roi_r=self.roi.right,
                                                     alpha=alpha, winsize=window_size, zoom=zoom,
                                                     sub_thresh=sub_thresh)

        # set the current fields
        # self.needle_shape = pts_3d[ :, 0:3 ]  # remove 4-th axis

        # self.img_points.left = pts_l
        # self.img_points.right = pts_r

        # self.img_bspline.left = pts_l
        # self.img_bspline.right = pts_r

        # self.processed_images = imgs
        # self.processed_figures = figs

        return imgs #pts_3d[ :, 0:3 ]

    # reconstruct_needle


# class:StereoNeedleRefReconstruction
