#!/usr/bin/env python
# coding: utf-8

#These functions are for image segmentation by using classical algorithm, watershed segmentation and image pre and postprocessing.

import numpy as np
import os, sys
import copy
import cv2
import mahotas as mh
from skimage.measure import label
from tqdm import tqdm
from skimage.morphology import opening, closing
from skimage.morphology import disk
from skimage.filters import thresshold_otsu, threshold_triangle, threshold_yen, threshold_mean, threshold_minimum, threshold_li, threshold_isodata
import itertools


class watershed_preprocessing:
    
    @staticmethod
    def labels_selection_by_area(image, area):
        """
        Functions for screening noise by area and returning that result.
        Area of each component is calculated by connected component function implemented in openccv
        
        Input
        ------
        image: 2D image
               binarized image
        area : int
               area threshold
               
        Return
        ------
        labels     : list
                     label list after screening
        labels_drop: list
                     eliminated labels by screening 
        
        
        """
        #labeling by connected components
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(image))
        
        #extract labels to be omitted
        label_drop=[i for i in range(len(stats)) if stats[i][4]<area]
        
        return labels, label_drop
    
    @staticmethod
    def labels_selection_by_area_and_aspect_ratio(image, area_1, area_2, aspect_ratio):
        """
        Funtions for screening noise by area and aspect ratio, and for returning that result.
        Area and aspect ration of each component is calculated by connected component function implemented in openccv
        
        Input
        ------
        image       : 2D image
                      input image
        area_1      : int
                      area thershold. This is used with aspect ratio threshold
        area_2      : int
                      area thershold
        aspect_ratio: float
                      aspect ratio thershold. This is used with area threshold
                      
        Return
        ------
        labels     : list
                     label list after screening
        labels_drop: list
                     eliminated labels by screening 
        
        """
        
        #labeling by connected components
        _, labels, stats, _ = cv2.connectedComponentsWithStats(image)

        #extract labels to be omitted
        label_drop_1=[i for i in range(len(stats)) if stats[i][4]<area_1 and stats[i][3]/stats[i][2]<aspect_ratio]
        label_drop_2=[i for i in range(len(stats)) if stats[i][4]<area_2]
        label_drop=list(set(list(itertools.chain(label_drop_1, label_drop_2))))
        
        return labels, label_drop
    
    @staticmethod
    def labels_selection_by_area_or_aspect_ratio(image, area, aspect_ratio):
        """
        Funtions for screening noise by area and aspect ratio, and for returning that result.
        Area and aspect ration of each component is calculated by connected component function implemented in openccv
        
        Input
        ------
        image       : 2D image
                      input image
        area        : int
                      area thershold
        aspect_ratio: float
                      aspect ratio thershold
        
        Return
        ------
        labels     : list
                     label list after screening
        labels_drop: list
                     eliminated labels by screening 
        
        """
        
        #labeling by connected components
        _, labels, stats, _ = cv2.connectedComponentsWithStats(image)

        #some componnents are droped by restriction on area and aspect ratio
        label_drop_1=[i for i in range(len(stats)) if stats[i][4]<area]
        label_drop_2=[i for i in range(len(stats)) if stats[i][3]/stats[i][2]>aspect_ratio]
        label_drop=list(set(list(itertools.chain(label_drop_1, label_drop_2))))
        
        return labels, label_drop
        
    
    def omit_by_area(self, image, area, inversion="True"):
        """
        Functions for screening noise by area
        
        Input
        ------
        image    : 2D image
                   input binarized image
        area     : int
                   area thresold
        inversion: boolean
                   If True, positive and negative inversion is performed before connected components
        
        Return
        ------
        labels: 2D image
                binarized image after noise elimination
        
        """
        
        #define image dimension
        height, width=image.shape
        
        if inversion=="True":
            #positive negative inversion
            image=cv2.bitwise_not(image)
        
        labels, label_drop=watershed_preprocessing.labels_selection_by_area(image, area)
        labels=labels.flatten()
        
        #pixels assigned to label_drop_lumen are set to 0 
        for i in tqdm(range(len(label_drop))):
            label_drop_ind=np.where(labels==label_drop[i])
            labels[label_drop_ind]=0
        
        labels=labels.reshape(height, width)
        labels=np.where(labels>0, 255, 0)
        
        return labels
    
    
    def omit_by_area_aspect_ratio(self, image, area_1, area_2, aspect_ratio, inversion="True"):
        """
        Functions for screening noise by area and aspect ratio 
        
        Input
        ------
        image       : 2D image
                      input binarized image
        area_1      : int
                      area thresold used with aspect ratio thresold
        area_2      : int
                      area threshold
        aspect_ratio: float
                      aspect ratio thershold used with area thershold
        inversion   : boolean
                      If True, positive and negative inversion is performed before connected components
        
        Return
        ------
        labels: 2D image
                binarized image after noise elimination
        
        """
        
        #define image dimension
        height, width=image.shape
        
        if inversion=="True":
            #positive negative inversion
            image=cv2.bitwise_not(np.uint8(image))

        labels, label_drop=watershed_preprocessing.labels_selection_by_area_and_aspect_ratio(image, area_1, area_2, aspect_ratio)
        labels=labels.flatten()
        
        for i in tqdm(range(len(label_drop))):
            label_drop_ind=np.where(labels==label_drop[i])
            labels[label_drop_ind]=0
        
        labels=labels.reshape(height, width)
        labels=np.where(labels>0, 255, 0)
  
        return labels
    
    
    def ray_extraction(self, image, kernel, iterations, area, aspect_ratio):
        """
        Function for selectively extracting ray region.
        Ray region is detectd by combination of morphological operation and screening by area and aspect ratio.
        
        Input
        ------
        image      : 2D image
                     input images
        kernel     : tuple
                     kernel for erosion morpphological operation
        iterations : int
                     iteration number for repeating erosion morphological operation
        area       : int
                     area threshold
        aspet_ratio: float
                     aspect ratio threshold
        
        Return
        ------
        ray_region: 2D image
                    binarized image (ray region is selectively extracted) 
        
        """
        
        #small regions coming from intermediate layer are omitted by morphological operator 
        height, width=image.shape
        
        kernel = np.ones(kernel ,np.uint8)
        erosion = cv2.erode(np.uint8(image), kernel, iterations = iterations)
        
        labels, label_drop=watershed_preprocessing.labels_selection_by_area_or_aspect_ratio(erosion, area, aspect_ratio)
        labels=labels.flatten()
        
        for i in tqdm(range(len(label_drop))):
            label_drop_ind=np.where(labels==label_drop[i])
            labels[label_drop_ind]=0
        
        label_map=np.where(labels.reshape(height, width)>0, 255, 0)
        gradient = cv2.morphologyEx(np.uint8(label_map), cv2.MORPH_GRADIENT, kernel)
        ray_region=np.where(label_map-gradient>200, 255, 0)
        
        return ray_region
    
    
class watershed:
    
    @staticmethod   
    def watershed_segmentation(image):
        """
        Function for applying watershed segmentation to input image.
        Original function to perform watershed segmentation is implemented in mahotas.
        
        Input
        ------
        image: 2D image
               input image
        
        Return
        ------
        nuclei: 2D image
                label map obtained by watershed
        lines : 2D image (boolean)
                boundary image obtained by watershed. pixel corresponding to boundary has True. 
                width of boundary line is set to 1 pixel.
        
        """
        locmax=mh.regmax(image)
        seeds, nr_nuclei = mh.label(locmax)

        #distans transformation
        T = mh.thresholding.otsu(np.uint8(image))
        dist = mh.distance(np.uint8(image) > T)
        dist = dist.max() - dist
        dist -= dist.min()
        dist = dist/float(dist.ptp()) * 255
        dist = dist.astype(np.uint8)
        
        nuclei, lines = mh.cwatershed(dist, seeds, return_lines=True)
        
        return nuclei, lines

    
class watershed_postprocessing:
    
    @staticmethod
    def boundary_overlay(original_image, boundary_image, kernel):
        """
        Functions for drawing boundary lines on original image.
        Thickenss of boundary can be controlled by kernel
        
        Input
        ------
        original_image: 2D image (one or three channel)
                        input image in watershed segmentation 
        boundary_image: 2D image
                        cell boundary image obtained by watershed
        kernel        : tuple
                        kernel for gaussian blur
        
        Return
        ------
        im_copy: 2D image (one or three channel)
                 returned original image on which boundary lines are drawn
        
        """
        im_dim=len(original_image.shape)
        
        if im_dim==2:
            height, width=original_image.shape
        if im_dim==3:
            height, width, channel=original_image.shape
        
        #blur boundary lines for good visualization
        blur = cv2.GaussianBlur(boundary_image, kernel, 0)
        blur = np.where(blur[:,:]>0, 255, 0)
        
        if im_dim==2:
            im_copy=np.zeros((height, width))
            im_copy[:,:]=original_image
            im_copy[np.where(blur[:,:]==255)]=[255, 0, 0]
            
            return im_copy
        
        if im_dim==3:
            im_copy=np.zeros((height, width, channel), np.uint8)
            im_copy[:,:, :]=original_image
            im_copy[np.where(blur[:,:]==255)]=[255, 0, 0]
            
            return im_copy
        
        else:
            print("Image dimension is invalid")
            
    @staticmethod
    def save_overlay_im(boundary_image, original_image, save_path, save_name):
        """
        Functions for automatically saving boundary and original images applicable to manual correction.
        Original image dimension is kept after saving.
        
        Input
        ------
        boundary_image: 2D image
                        binary image on which cell boundaries are drawn.
                        Cell boundaries and th other bakground correspond to 255 and 0.
        original_image: 2D image (1 or 3 channel)
                        original image applied to segmentation
        save_path     : path
                        path for saving results
        save_name     : str
                        image name for saving (without extention).
                        Image save format is fixed to .png
        
        """
        
        if len(boundary_image.shape)>2:
            print("Image dimension is invalid")
            return boundary_image
        
        else:
            boundary_image_tr=np.zeros((boundary_image.shape[0], boundary_image.shape[1], 4), np.uint8)
            boundary_image_tr[:,:,2]=boundary_image
            boundary_image_tr[:,:,3] = np.where(np.all(boundary_image_tr == 0, axis=-1), 0, 255) 

            #set save path
            if os.path.exists(save_path)==False:
                os.makedirs(save_path)

            cv2.imwrite(os.path.join(save_path, 'boundary_for_mannual_correction_'+str(save_name)+'.png'), boundary_image_tr)
            cv2.imwrite(os.path.join(save_path, 'ground_im_for_manual_correction_'+str(save_name)+'.png'), original_image)
        