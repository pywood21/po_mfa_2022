#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import imutils
import copy
from skimage.filters import threshold_otsu


"""
Below class function contains two algorithm for image registration, named SIFT [1] and POC [2].
In our paper, POC technique is used for image concatenation of retardation images, and FLM micrographs to restore one large image. 
For calling this class, opencv-contrib must be installed.

Refernce
[1] Lowe DG. 2004. Distinct image features from scale-invariant keypoints. International Journal of Computer Vision 60, 91-110.
[2] Ito K, Nakajima H, Kobayashi K, Aoki T, Higuchi T. 2004. A fingerprint matching algorithm using phase-only correlation. IEICE Transactions on Fundamental Electronics, Communications and Computer Sciences E87-A, 682-691.

"""

class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()
        
       
    ### SIFT algorithm ###
    
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False, overlap=False, ori_size=True):
        """
        Function for image registration using SIFT and for returning that result.
        Details of functions are decripted in opencv reference.
        
        Input
        ------
        images      : list of 2D image
                      image pair to be stitched. A pair of images must have same dimensions.
        ratio       : float
                      threshold for ensuring that the distance is within a certain ratio of each
        reprojThresh: float
                      maximum value of inliers. 
        showMatches : boolean (True or False)
                      If True, image visualzing keypoint matching results is returned. 
        ori_size    : boolean (True or False)
                      If True, image size after sticthing keeps the same dimension before sticthing.  
        
        
        Return
        ------
        image_left   : 2D image
                       result image after sticthing
        vis          : 2D image
                       image visualizing keypoint matching
        delete_line_h: list
                       pixel indices in row direction belonging to bachground created during stitching
        delete_line_v: list
                       pixel indices in column direction belonging to bachground created during stitching
        H            : matrix
                       transformation matrix used in stitching.
        
        """
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images 
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
 
        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
            featuresA, featuresB, ratio, reprojThresh)
 
        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None
        
        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        
        
        #define the image size after image sticthing
        #if ori_size==True, size of 
        if ori_size==True:
            result = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0]))
        
            ### delete background
            delete_line = np.where(np.all(result == 0, axis=0))[0]
            delete_line_h = np.where(np.all(result == 0, axis=1))[0]
            delete_line_1 = delete_line[delete_line>imageA.shape[1]]
            delete_line_v = delete_line[delete_line<imageA.shape[1]]
            image_left = result
        else:
            result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        
            ### delete background
            delete_line = np.where(np.all(result == 0, axis=0))[0]
            delete_line_h = np.where(np.all(result == 0, axis=1))[0]
            delete_line_1 = delete_line[delete_line>imageA.shape[1]]
            delete_line_v = delete_line[delete_line<imageA.shape[1]]
            image_left = np.delete(result, delete_line, axis=1)
            
            
        if overlap==True:
        
            coverred_part = image_left.shape[1]//2 + 1 
            image_left[0:imageB.shape[0], 0:coverred_part] = imageB[:, 0:coverred_part] 
        
        print("Size of original image, imageA: "+str(imageA.shape[0])+"*"+str(imageA.shape[1]))
        print("Size of original image, imageB: "+str(imageB.shape[0])+"*"+str(imageB.shape[1]))
        print("After concatenation, image: "+str(image_left.shape[0])+"*"+str(image_left.shape[1]))
        
        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
 
            # return a tuple of the stitched image and the
            # visualization
            return (image_left, vis), delete_line_h, delete_line_v, H
        
        else:
            # return the stitched image
            return image_left,  delete_line_h, delete_line_v, H
    
    
    def detectAndDescribe(self, image):
        """
        Function for image feature detection by SIFT algorithm and for returning them.
        Details of functions are decripted in opencv reference.
        
        Input
        ------
        image: 2D image
               image from which you would like to keypoints
        
        Return
        ------
        kps     : array
                  array of keypoints object
        features: array
                  array of descriptors
        
        """
        
        #call descriptor
        descriptor=cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image,None)
 
        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])
 
        # return a tuple of keypoints and features
        return (kps, features)
    
 
    ### matching detected features
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        """
        Function for image registration using keypoints detected by SIFT algorithm.
        Details of functions are decripted in opencv reference.
        
        Input
        ------
        kpsA, kpsB          : array
                              array of keypoints
        featuresA, featuresB: array
                              array of descriptors
        ratio               : float
                              threshold for ensuring that the distance is within a certain ratio of each
        reprojThresh        : float
                              maximum value of inliers. 
        
        
        Return
        ------
        matches: list
                 pair of matched kepoint features obtained from two images
        H      : matrix
                 transformation matrix
        status : mask
        
        """
        
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
 
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
                
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
 
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
 
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)
        
        else:
            # otherwise, no homograpy could be computed
            return None

    
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        """
        Function for visualizing keypoint matchings.
        
        Input
        ------
        kpsA, kpsB: array
                    array of keypoints
        matches   : list
                    pair of matched kepoint features obtained from two images
        status    : mask
        
        Return
        ------
        vis: 2D image
             image for visualizing keypoint matching
        
        """
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB), dtype="uint8") ############
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
 
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
 
        # return the visualization
        return vis
    
    
    ### POC algorithm ###
    
    def POC_reference_criterion(self, criterion_image_list, target_image_list):
        """
        Image concatenation by POC algorithm.
        Two kinds of image sets are assumed as your inputs (criterion_image_list and target_image_list).
        First pair is regarded as dummy image sets for finding out the best image concatenation conditions.
        Generally, binarized images works well as this role.
        Based on the found condition, your main target image pairs are concatenated.
        This function is applicable to 1 channel and multi-channel images.
        
        Input
        ------
        criterion_image_list: list of images
                              criterion images to be sticthed
        target_image_list   : list of images
                              target images to be sticthed
        Return
        ------
        image1_criterion: 2D image
                          reference image after image sticthing
        image1_target   : 2D image
                          target image after image stitching
        
        """
        
        #set while loop
        i=0
        while(True):    
            #define image pair for stitching
            if i==0:
                image1_criterion=criterion_image_list[i]
                image1_target=target_image_list[i]
            image2_criterion=criterion_image_list[i+1]
            image2_target=target_image_list[i+1]
            
            #check target image dimension
            im_dim=len(image1_target.shape)
            
            if im_dim==2:
                height, width=image1_target.shape

                #adjust image dimension
                if image1_criterion.shape!=image2_criterion.shape:
                    image2_criterion_copy=np.zeros((image1_criterion.shape[0], image1_criterion.shape[1]))
                    image2_target_copy=np.zeros((image1_target.shape[0], image1_target.shape[1]))
                    height_temp, width_temp=image2_criterion_copy.shape

                    #original image is set to the center
                    image2_criterion_copy[height_temp//2-image2_criterion.shape[0]//2:height_temp//2+image2_criterion.shape[0]//2,
                                width_temp//2-image2_criterion.shape[1]//2:width_temp//2+image2_criterion.shape[1]//2]=image2_criterion
                    image2_target_copy[height_temp//2-image2_target.shape[0]//2:height_temp//2+image2_target.shape[0]//2,
                                width_temp//2-image2_target.shape[1]//2:width_temp//2+image2_target.shape[1]//2]=image2_target
                    
                    image2_criterion=image2_criterion_copy
                    image2_target=image2_target_copy
            
            if im_dim==3:
                height, width, channel=image1_target.shape
                
                #adjust image dimension
                if image1_criterion.shape!=image2_criterion.shape:
                    image2_criterion_copy=np.zeros((image1_criterion.shape[0], image1_criterion.shape[1]))
                    image2_target_copy=np.zeros((image1_target.shape[0], image1_target.shape[1], channel), np.unit8)
                    height_temp, width_temp=image2_criterion_copy.shape

                    #original image is set to the center
                    image2_criterion_copy[height_temp//2-image2_criterion.shape[0]//2:height_temp//2+image2_criterion.shape[0]//2,
                                width_temp//2-image2_criterion.shape[1]//2:width_temp//2+image2_criterion.shape[1]//2]=image2_criterion
                    image2_target_copy[height_temp//2-image2_target.shape[0]//2:height_temp//2+image2_target.shape[0]//2,
                                width_temp//2-image2_target.shape[1]//2:width_temp//2+image2_target.shape[1]//2, :]=image2_target
                    
                    image2_criterion=image2_criterion_copy
                    image2_target=image2_target_copy
            
            
            #apply poc. template is image1
            d, etc = cv2.phaseCorrelate(np.float32(image2_criterion), np.float32(image1_criterion))

            #calculated position shift 
            x_abs_shift=int(np.abs(d[0]))
            y_abs_shift=int(np.abs(d[1]))

            #set zero matrix
            if im_dim==2:
                im_zero_1_criterion=np.zeros((height+2*y_abs_shift, width+2*x_abs_shift))
                im_zero_2_criterion=np.zeros((height+2*y_abs_shift, width+2*x_abs_shift))
                im_zero_1_target=np.zeros((height+2*y_abs_shift, width+2*x_abs_shift))
                im_zero_2_target=np.zeros((height+2*y_abs_shift, width+2*x_abs_shift))
                
                #fill zero matrix by original image
                im_zero_1_criterion[y_abs_shift:height+y_abs_shift, x_abs_shift:width+x_abs_shift]=image1_criterion
                im_zero_2_criterion[y_abs_shift:height+y_abs_shift, x_abs_shift:width+x_abs_shift]=image2_criterion
                im_zero_1_target[y_abs_shift:height+y_abs_shift, x_abs_shift:width+x_abs_shift]=image1_target
                im_zero_2_target[y_abs_shift:height+y_abs_shift, x_abs_shift:width+x_abs_shift]=image2_target
                
                
            if im_dim==3:
                im_zero_1_criterion=np.zeros((height+2*y_abs_shift, width+2*x_abs_shift))
                im_zero_2_criterion=np.zeros((height+2*y_abs_shift, width+2*x_abs_shift))
                im_zero_1_target=np.zeros((height+2*y_abs_shift, width+2*x_abs_shift, channel), np.uint8)
                im_zero_2_target=np.zeros((height+2*y_abs_shift, width+2*x_abs_shift, channel), np.uint8)
                
                #fill zero matrix by original image
                im_zero_1_criterion[y_abs_shift:height+y_abs_shift, x_abs_shift:width+x_abs_shift]=image1_criterion
                im_zero_2_criterion[y_abs_shift:height+y_abs_shift, x_abs_shift:width+x_abs_shift]=image2_criterion
                im_zero_1_target[y_abs_shift:height+y_abs_shift, x_abs_shift:width+x_abs_shift, :]=image1_target
                im_zero_2_target[y_abs_shift:height+y_abs_shift, x_abs_shift:width+x_abs_shift, :]=image2_target

                
            #affine transformation
            M = np.float32([[1, 0, d[0]],[0, 1, d[1]]])
            img_result_criterion = cv2.warpAffine(np.float32(im_zero_2_criterion), M, (width+2*x_abs_shift, height+2*y_abs_shift), 
                                                 flags=cv2.INTER_NEAREST)
            img_result_target = cv2.warpAffine(np.float32(im_zero_2_target), M, (width+2*x_abs_shift, height+2*y_abs_shift), 
                                              flags=cv2.INTER_NEAREST)
       
            #img_result[y_abs_shift:height+y_abs_shift, x_abs_shift:width+x_abs_shift]=image1
            img_result_criterion=np.where(im_zero_1_criterion>0, im_zero_1_criterion, img_result_criterion)
            img_result_target=np.where(im_zero_1_target>0, im_zero_1_target, img_result_target)
           
            #delete background region
            delete_line_v = np.where(np.all(img_result_criterion == 0, axis=0))[0]
            delete_line_h = np.where(np.all(img_result_criterion == 0, axis=1))[0]
            img_result_criterion = np.delete(img_result_criterion, delete_line_v, axis=1)
            img_result_criterion = np.delete(img_result_criterion, delete_line_h, axis=0)
            img_result_target = np.delete(img_result_target, delete_line_v, axis=1)
            img_result_target = np.delete(img_result_target, delete_line_h, axis=0)
            
            if im_dim==3:
                img_result_target=np.uint8(img_result_target)

            #set result image as image1
            threshold = threshold_otsu(img_result_criterion)
            img_result_criterion = np.where(img_result_criterion < threshold, 0, 255)
            
            #stitched result is set to image1 for next loop
            image1_criterion=img_result_criterion
            image1_target=img_result_target
            
            i+=1

            if i==len(criterion_image_list)-1:
                break
    
        return image1_criterion, image1_target
    
    
    def image_dimension_adj(self, imageA, imageB, padding, criterion="A"):
        """
        Function for image dimension adjustment of two images with different dimensions (only applicable to 1 channel image).
        This function must be required to use image registration by SIFT and POC in this class function when your target images have diffrent sizes.
        
        Input
        ------
        imageA, imageB: 2D image
                        input images to be adjusted in their size
        padding       : int
                        interger values for padding margin of images after image resizing
        criterion     : "A", "B" or "AB".
                        image size adjustment criterion. 
                        If "A", dimension of imageB is adjusted to that of imageA. If "B", dimension of imageA is adjusted to that of imageB.
                        If "AB", the longer image height and width are selected as image dimension after adjustment.
        
        Return
        ------
        imageA_: 2D image
                 imageA after size adjustment
        imageB_: 2D image
                 imageB after size adjustment
        
        """
        
        height_A, width_A=imageA.shape
        height_B, width_B=imageB.shape
        print("criterion="+str(criterion))
        
        #when criterion is equal to 0, dimension of imageB is reset to that of imageA.
        #when criterion is equal to 1, dimension of imageA is reset to that of imageB.
        #when criterion is not defined, dimension of each images are adjusted to the longer height and width.
        if criterion=="A":
            imageA_=imageA
            imageB_=np.zeros((height_A, width_A))
            
            if height_A>height_B:
                if width_A>width_B:
                    imageB_[:height_B, :width_B]=imageB[:height_B, :width_B]
                if width_B>width_A:
                    imageB_[:height_B, :width_A]=imageB[:height_B, :width_A]
                    imageB_[height_B:]=padding #margin areas are replaced by padding values
                else:
                    imageB_[:height_B, :width_B]=imageB[:height_B, :width_B]
                    imageB_[height_B:, width_B:]=padding #margin areas are replaced by padding values
            if height_B>height_A:
                if width_A>width_B:
                    imageB_[:height_A, :width_B]=imageB[:height_A, :width_B]
                    imageB_[:, width_B:]=padding #margin areas are replaced by padding values
                if width_B>width_A:
                    imageB_[:height_A, :width_A]=imageB[:height_A, :width_A]
                else:
                    imageB_[:height_A, :width_B]=imageB[:height_A, :width_B]
                    imageB_[:, width_B:]=padding #margin areas are replaced by padding values
            if height_A==height_B:
                if width_A>width_B:
                    imageB_[:height_A, :width_B]=imageB[:height_A, :width_B]
                    imageB_[:, width_B:]=padding #margin areas are replaced by padding values
                if width_B>width_A:
                    imageB_[:height_A, :width_A]=imageB[:height_A, :width_A]
                else:
                    imageB_[:height_A, :width_B]=imageB[:height_A, :width_B]
                 
            
        if criterion=="B":
            imageA_=np.zeros((height_B, width_B))
            imageB_=imageB
            
            if height_A>height_B:
                if width_A>width_B:
                    imageA_[:height_B, :width_B]=imageA[:height_B, :width_B]
                if width_B>width_A:
                    imageA_[:height_B, :width_A]=imageA[:height_B, :width_A]
                    imageA_[:, width_A:]=padding #margin areas are replaced by padding values
                else:
                    imageA_[:height_B, :width_B]=imageA[:height_B, :width_B]
            if height_B>height_A:
                if width_A>width_B:
                    imageA_[:height_A, :width_B]=imageA[:height_A, :width_B]
                    imageA_[height_A:]=padding
                if width_B>width_A:
                    imageA_[:height_A, :width_A]=imageA[:height_A, :width_A]
                    imageA_[height_A:, width_A:]=padding #margin areas are replaced by padding values
                else:
                    imageA_[:height_A, :width_B]=imageA[:height_A, :width_B]
                    imageA_[height_A:]=padding
            if height_A==height_B:
                if width_A>width_B:
                    imageA_[:height_A, :width_B]=imageA[:height_A, :width_B]
                if width_B>width_A:
                    imageA_[:height_A, :width_A]=imageA[:height_A, :width_A]
                    imageA_[height_A:, width_A:]=padding #margin areas are replaced by padding values
                else:
                    imageA_[:height_A, :width_B]=imageA[:height_A, :width_B]
            
        
        if criterion=="AB":
            #set list
            height_list=[height_A, height_B]
            width_list=[width_A, width_B]

            #max val is extracted
            height_maxind=np.argmax(height_list)
            width_maxind=np.argmax(width_list)
            height_minind=np.argmin(height_list)
            width_minind=np.argmin(width_list)

            if height_maxind==height_minind:
                if width_maxind==width_minind:
                    imageA_=imageA
                    imageB_=imageB
                else:
                    imageA_=np.zeros((height_list[height_maxind], width_list[width_maxind]))
                    imageB_=np.zeros((height_list[height_maxind], width_list[width_maxind]))
                    imageA_[:height_A, :width_A]=imageA
                    imageB_[:height_B, :width_B]=imageB
            else:
                imageA_=np.zeros((height_list[height_maxind], width_list[width_maxind]))
                imageB_=np.zeros((height_list[height_maxind], width_list[width_maxind]))
                imageA_[:height_A, :width_A]=imageA
                imageB_[:height_B, :width_B]=imageB
             
        
        height_A_, width_A_=imageA_.shape
        height_B_, width_B_=imageB_.shape
        
        #show results
        print("imageA: height="+str(height_A)+", width="+str(width_A))
        print("imageB: height="+str(height_B)+", width="+str(width_B))
        print("After adjustment, imageA: height="+str(height_A_)+", width="+str(width_A_))
        print("After adjustment, imageB: height="+str(height_B_)+", width="+str(width_B_))
        
        return imageA_, imageB_
    
    
    def image_dimension_adj_const(self, imageA, imageB, padding, const, criterion="A"):
        """
        Function for image dimension adjustment of two images with different dimensions (only applicable to 1 channel image).
        Background pixel values can be controlled by const.
        This function must be required to use image registration by SIFT and POC in this class function when your target images have diffrent sizes.
        
        Input
        ------
        imageA, imageB: 2D image
                        input images to be adjusted in their size
        padding       : int
                        interger values for padding margin of images after image reisizing
        const         : int
                        background pixel values
        criterion     : "A", "B" and "AB".
                        If "A", dimension of imageB is adjusted to that of imageA. If "B", dimension of imageA is adjusted to that of imageB.
                        If "AB", the longer image height and width are selected as image dimension after adjustment.
        
        Return
        ------
        imageA_: 2D image
                 imageA after size adjustment
        imageB_: 2D image
                 imageB after size adjustment
        
        """
        height_A, width_A=imageA.shape
        height_B, width_B=imageB.shape
        print("criterion="+str(criterion))
        
        #if criterion is "A", below codes will run.
        if criterion=="A":
            imageA_=imageA
            imageB_=np.ones((height_A, width_A))*const
            
            if height_A>height_B:
                if width_A>width_B:
                    imageB_[:height_B, :width_B]=imageB[:height_B, :width_B]
                if width_B>width_A:
                    imageB_[:height_B, :width_A]=imageB[:height_B, :width_A]
                    imageB_[height_B:]=padding
                else:
                    imageB_[:height_B, :width_B]=imageB[:height_B, :width_B]
                    imageB_[height_B:, width_B:]=padding
            if height_B>height_A:
                if width_A>width_B:
                    imageB_[:height_A, :width_B]=imageB[:height_A, :width_B]
                    imageB_[:, width_B:]=padding
                if width_B>width_A:
                    imageB_[:height_A, :width_A]=imageB[:height_A, :width_A]
                else:
                    imageB_[:height_A, :width_B]=imageB[:height_A, :width_B]
                    imageB_[:, width_B:]=padding
            if height_A==height_B:
                if width_A>width_B:
                    imageB_[:height_A, :width_B]=imageB[:height_A, :width_B]
                    imageB_[:, width_B:]=padding
                if width_B>width_A:
                    imageB_[:height_A, :width_A]=imageB[:height_A, :width_A]
                else:
                    imageB_[:height_A, :width_B]=imageB[:height_A, :width_B]
                 
        #if criterion is "B", below codes will run.
        if criterion=="B":
            imageA_=np.ones((height_B, width_B))*const
            imageB_=imageB
            
            if height_A>height_B:
                if width_A>width_B:
                    imageA_[:height_B, :width_B]=imageA[:height_B, :width_B]
                if width_B>width_A:
                    imageA_[:height_B, :width_A]=imageA[:height_B, :width_A]
                    imageA_[:, width_A:]=padding
                else:
                    imageA_[:height_B, :width_B]=imageA[:height_B, :width_B]
            if height_B>height_A:
                if width_A>width_B:
                    imageA_[:height_A, :width_B]=imageA[:height_A, :width_B]
                    imageA_[height_A:]=padding
                if width_B>width_A:
                    imageA_[:height_A, :width_A]=imageA[:height_A, :width_A]
                    imageA_[height_A:, width_A:]=padding
                else:
                    imageA_[:height_A, :width_B]=imageA[:height_A, :width_B]
                    imageA_[height_A:]=padding
            if height_A==height_B:
                if width_A>width_B:
                    imageA_[:height_A, :width_B]=imageA[:height_A, :width_B]
                if width_B>width_A:
                    imageA_[:height_A, :width_A]=imageA[:height_A, :width_A]
                    imageA_[height_A:, width_A:]=padding
                else:
                    imageA_[:height_A, :width_B]=imageA[:height_A, :width_B]
            
        #if criterion is "AB", below codes will run.
        if criterion=="AB":
            #set list
            height_list=[height_A, height_B]
            width_list=[width_A, width_B]

            #max val is extracted
            height_maxind=np.argmax(height_list)
            width_maxind=np.argmax(width_list)
            height_minind=np.argmin(height_list)
            width_minind=np.argmin(width_list)

            if height_maxind==height_minind:
                if width_maxind==width_minind:
                    imageA_=imageA
                    imageB_=imageB
                else:
                    imageA_=np.ones((height_list[height_maxind], width_list[width_maxind]))*const
                    imageB_=np.ones((height_list[height_maxind], width_list[width_maxind]))*const
                    imageA_[:height_A, :width_A]=imageA
                    imageB_[:height_B, :width_B]=imageB
            else:
                imageA_=np.ones((height_list[height_maxind], width_list[width_maxind]))*const
                imageB_=np.ones((height_list[height_maxind], width_list[width_maxind]))*const
                imageA_[:height_A, :width_A]=imageA
                imageB_[:height_B, :width_B]=imageB
             
        
        height_A_, width_A_=imageA_.shape
        height_B_, width_B_=imageB_.shape
        
        #return image dimension before and after image registration
        print("imageA: height="+str(height_A)+", width="+str(width_A))
        print("imageB: height="+str(height_B)+", width="+str(width_B))
        print("After adjustment, imageA: height="+str(height_A_)+", width="+str(width_A_))
        print("After adjustment, imageB: height="+str(height_B_)+", width="+str(width_B_))
        
        return imageA_, imageB_
    
    
    def image_dimension_adj_3D(self, imageA, imageB, padding, criterion="A"):
        """
        Function for image dimension adjustment of two images with different dimensions (only applicable to 3 channel image).
        This function must be required to use image registration by SIFT and POC in this class function when your target images have diffrent sizes.
        
        Input
        ------
        imageA, imageB: 2D image
                        input images to be adjusted in their size
        padding       : int
                        interger values for padding margin of images after image reisizing
        const         : int
                        background pixel values
        criterion     : "A", "B" and "AB".
                        If "A", dimension of imageB is adjusted to that of imageA. If "B", dimension of imageA is adjusted to that of imageB.
                        If "AB", the longer image height and width are selected as image dimension after adjustment.
        
        Return
        ------
        imageA_: 2D image
                 imageA after size adjustment
        imageB_: 2D image
                 imageB after size adjustment
        
        """
        
        height_A, width_A, channel_A=imageA.shape
        height_B, width_B, channel_B=imageB.shape
        print("criterion="+str(criterion))
        
        #if criterion is "A", below codes will run.
        if criterion=="A":
            imageA_=imageA
            imageB_=np.zeros((height_A, width_A, channel_A), np.uint8)
            
            if height_A>height_B:
                if width_A>width_B:
                    imageB_[:height_B, :width_B, :]=imageB[:height_B, :width_B, :]
                if width_B>width_A:
                    imageB_[:height_B, :width_A, :]=imageB[:height_B, :width_A, :]
                    imageB_[height_B:, :, :]=padding
                else:
                    imageB_[:height_B, :width_B, :]=imageB[:height_B, :width_B, :]
                    imageB_[height_B:, width_B:, :]=padding #margin areas are replaced by padding values
            if height_B>height_A:
                if width_A>width_B:
                    imageB_[:height_A, :width_B, :]=imageB[:height_A, :width_B, :]
                    imageB_[:, width_B:, :]=padding
                if width_B>width_A:
                    imageB_[:height_A, :width_A, :]=imageB[:height_A, :width_A, :]
                else:
                    imageB_[:height_A, :width_B, :]=imageB[:height_A, :width_B, :]
                    imageB_[:, width_B:, :]=padding #margin areas are replaced by padding values
            if height_A==height_B:
                if width_A>width_B:
                    imageB_[:height_A, :width_B, :]=imageB[:height_A, :width_B, :]
                    imageB_[:, width_B:, :]=padding
                if width_B>width_A:
                    imageB_[:height_A, :width_A, :]=imageB[:height_A, :width_A, :]
                else:
                    imageB_[:height_A, :width_B, :]=imageB[:height_A, :width_B, :]
                 
        #if criterion is "B", below codes will run.
        if criterion=="B":
            imageA_=np.zeros((height_B, width_B, channel_B), np.uint8)
            imageB_=imageB
            
            if height_A>height_B:
                if width_A>width_B:
                    imageA_[:height_B, :width_B, :]=imageA[:height_B, :width_B, :]
                if width_B>width_A:
                    imageA_[:height_B, :width_A, :]=imageA[:height_B, :width_A, :]
                    imageA_[:, width_A:, :]=padding #margin areas are replaced by padding values
                else:
                    imageA_[:height_B, :width_B, :]=imageA[:height_B, :width_B, :]
            if height_B>height_A:
                if width_A>width_B:
                    imageA_[:height_A, :width_B, :]=imageA[:height_A, :width_B, :]
                    imageA_[height_A:, :, :]=padding
                if width_B>width_A:
                    imageA_[:height_A, :width_A, :]=imageA[:height_A, :width_A, :]
                    imageA_[height_A:, width_A:, :]=padding
                else:
                    imageA_[:height_A, :width_B, :]=imageA[:height_A, :width_B, :]
                    imageA_[height_A:, :, :]=padding #margin areas are replaced by padding values
            if height_A==height_B:
                if width_A>width_B:
                    imageA_[:height_A, :width_B, :]=imageA[:height_A, :width_B, :]
                if width_B>width_A:
                    imageA_[:height_A, :width_A, :]=imageA[:height_A, :width_A, :]
                    imageA_[height_A:, width_A:, :]=padding #margin areas are replaced by padding values
                else:
                    imageA_[:height_A, :width_B, :]=imageA[:height_A, :width_B, :]
            
        #if criterion is "AB", below codes will run.
        if criterion=="AB":
            #set list
            height_list=[height_A, height_B]
            width_list=[width_A, width_B]

            #max val is extracted
            height_maxind=np.argmax(height_list)
            width_maxind=np.argmax(width_list)
            height_minind=np.argmin(height_list)
            width_minind=np.argmin(width_list)

            if height_maxind==height_minind:
                if width_maxind==width_minind:
                    imageA_=imageA
                    imageB_=imageB
                else:
                    imageA_=np.zeros((height_list[height_maxind], width_list[width_maxind], channel_A), np.uint8)
                    imageB_=np.zeros((height_list[height_maxind], width_list[width_maxind], channel_B), np.uint8)
                    imageA_[:height_A, :width_A, :]=imageA
                    imageB_[:height_B, :width_B, :]=imageB
            else:
                imageA_=np.zeros((height_list[height_maxind], width_list[width_maxind], channel_A), np.uint8)
                imageB_=np.zeros((height_list[height_maxind], width_list[width_maxind], channel_B), np.uint8)
                imageA_[:height_A, :width_A, :]=imageA
                imageB_[:height_B, :width_B, :]=imageB
             
        
        height_A_, width_A_, channel_A_=imageA_.shape
        height_B_, width_B_, channel_B_=imageB_.shape
        
        #return image dimension before and after image registration
        print("imageA: height="+str(height_A)+", width="+str(width_A))
        print("imageB: height="+str(height_B)+", width="+str(width_B))
        print("After adjustment, imageA: height="+str(height_A_)+", width="+str(width_A_))
        print("After adjustment, imageB: height="+str(height_B_)+", width="+str(width_B_))
        
        return imageA_, imageB_
    
