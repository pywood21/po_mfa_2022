#!/usr/bin/env python
# coding: utf-8

"""
This code is for extracting radial files.
Original idea is proposed by Brunel G et al.(2014) [1]
If you want to extract those with annual ring width, please use class Radial_file_annual_ring, additionally.

Reference
[1] Brunel G, Borianne P, Subsol G, Jaeger M, Caraglio Y. 2014. Automatic identification and characterization of radial files in light microscopy images of wood. Annals of Botany 114, 829-840.

"""

import matplotlib.pyplot as plt
import numpy as np
import os, sys
import cv2
import copy
from tqdm import tqdm
from skimage.future import graph
from skimage import draw
import itertools
from scipy import interpolate
import networkx as nx



class Radial_file_extraction:
    """
    This class contains fundamental functions for extracting radial files in softwoods utilizing RAG network.]
    
    """
    
    def __init__(self, label_map, cx_list, cy_list):
        """
        member
        ------
        label_map: 2D image.
                   2D cell label image. labels are assigned by instance segmentation like watershed. 
        cx_list  : list
                   list of x coordinates of cell centroids
        cy_list  : list
                   list of y coordinates of cell centroids
        
        """
        self.label_map=label_map
        self.cx_list=cx_list
        self.cy_list=cy_list
    
    
    def draw_RAG(self, image, return_im=False):
        """
        Funtion for drawing RAG network [2] on label image.
        RAG is calculated by scikit-image function, skimage.future.graph (originally implemented in networkx package).
        
        Reference
        [2] Trémaeu A, Colantoni P. 2000. Regions adjacency graph applied to color image segmentation. IEEE Transactions on Image Processing 9, 
            735-744. 
            
        Input
        ------
        image    : 2D image
                   Origin of label image (2D).
        return_im: boolean (True or False)
                   If True, RAG network is drawn on your input image and return it.
        
        Return
        ------
        edge_list: list
                   edge list of RAG network.
        image_inv: 2D image
                   RAG network results drawn on your input image. For contrast enhancement, positive and negative inversion is performed.
        
        """
        
        #draw RAG
        g = graph.rag_mean_color(image, self.label_map)
        
        #image is None, return only edge_list
        if return_im==False:
            edge_list=[]
            for edge in g.edges.items():
                n1, n2 = edge[0]
                if (n1==0) or (n2==0):
                    pass
                else:
                    edge_list.append(edge[0])
                    x1 = self.cx_list[n1]
                    y1 = self.cy_list[n1]
                    x2 = self.cx_list[n2]
                    y2 = self.cy_list[n2]
                    
            return edge_list
        
        #In the other cases, return edge_list and visualized result
        else:
            #visualize the result on original OM
            image_inv = cv2.bitwise_not(image)
            edge_list=[]
            for edge in g.edges.items():
                n1, n2 = edge[0]
                if (n1==0) or (n2==0):
                    pass
                else:
                    edge_list.append(edge[0])
                    x1 = self.cx_list[n1]
                    y1 = self.cy_list[n1]
                    x2 = self.cx_list[n2]
                    y2 = self.cy_list[n2]

                    line  = draw.line(y1, x1, y2, x2)
                    circle = draw.circle(y1,x1, 7)


                    image_inv[line] = 1
                    image_inv[circle] = 1

            return edge_list, image_inv
    
    
    @staticmethod
    def index_sort(index, j):
        if index[0]==j:
            return index[1]
        if index[1]==j:
            return index[0]
        else:
            pass

    #Bray-Curtis criterion
    @staticmethod
    def cost_func(index_1, index_2, param_list):
        """
        Functions for calculating Bray-Curtis index [3].
        If Bray-Curtis index of a certain anatomical parameter between two cells is small, those cells cen be considered as anatomically similar 
        ones.
        
        Refernce
        [3] Bray JR, Curtis JT. 1957. An ordination of the upland forest communities of southern Wisconsin. Ecological Monographs 27, 327-349.
        
        Input
        -----
        index_1   : int
                    index of the first cell
        index_2   : int
                    index of the second cell
        param_list: list
                    list of anatomical parameters such as cell transverse area.
        
        Return
        ------
        GS: float
            Bray-Curtis index of the anatomical parameter
        
        """
        
        GS=np.abs(param_list[index_1]-param_list[index_2])/(param_list[index_1]+param_list[index_2])
        
        return GS

    
    #radial file extraction
    #please customize cost and angle parameters
    #Background region should be set to 0 in your label map because label zero is skipped in below function
    
    def radial_file_extraction(self, anatomical_param_list, cost_list, angle_thres, edge_list):
        """
        Functions for radial file extraction applicable to softwoods.
        The function can find out probable radial files by sequential extraction of the most probable adjacent cells from the present cell.
        This can be achieved by two steps: 
        First step is screening by angle defined by horizontal line (parallel to radial direction) and vector connecting the present cell and its 
        adjacent cells. This step hypothesize that radial files run horizontally in general. Therefore, the most probable cell have the lower angle 
        if following this hypothesis. 
        Second step is screening by adjacent cell similarities. Bray-Curtis index enable us to calculate cell similarities by various kinds of 
        anatomical parameters in completely different unit. In this function, multiple anatomical parameters are available for this screening step.
        
        
        Input
        ------
        anatomical_param_list: list
                               list of anatomical parameters such as cell transverse area.
                               If you use multiple criterions based on various anatomical parameters, please input multuiple anatomical parameters 
                               as list in list. 
                               For example, if three anatomical parameters are used, 
                               anatomical_param_list = [[a0, ...., an], [b0, ...., bn], [c0, ...., cn]].
        cost_list            : list
                               list of cost threshold for screening by anatomical parameters (ranging from 0 to 1). 
                               If you use multiple criterions based on various anatomical parameters, please input multuiple cost as list.
                               For example, if three anatomical parameters are used, 
                               anatomical_param_list = [a_cost, b_cost, c_cost].
        angle_thres          : float (0-90)
                               angle threshold for screening. Ranging from 0 to 90. Angle thresold close to 0 means severe screening condition. 
        edge_list            : list
                               edge list extracted from RAG network.
        
        Return
        ------
        radial_file_label_list_summary: result of radial file extraction. Return is lists in list. Each list in list correspond to label sequences 
                                        consisting a radial file. 
        
        """
        
        #flatten label_map
        label_map=self.label_map.flatten()
        
        #rearange the label list based on x-coordinate of the center of gravity of each label in the manner of descending.
        cx_descend_list=self.cx_list[1:][np.argsort(self.cx_list[1:])[::-1]]
        cy_descend_list=self.cy_list[1:][np.argsort(self.cx_list[1:])[::-1]]
        label_descend_list=np.unique(label_map)[1:][np.argsort(self.cx_list[1:])[::-1]]

        #set the empty list saving radial file information
        radial_file_label_list_summary=[]
        cell_assignment_list=[]
        
        #angle restriction is set
        angle_thres=angle_thres
        
        #Plausible radial files are extarcted in below while loop until all cells belonge to a certain file 
        while(True):

            #breaking condition. if label_descend_list contains no labels, break loop.
            if len(label_descend_list)==0:
                break

            #set present_label
            present_label=label_descend_list[0]
            #update radial_file_label_list
            radial_file_label_list=[]
            
            #This loop corresponds to extract one radial file.
            while(True):
                #store the information of already extracted cell
                cell_assignment_list.append(present_label)

                #store the information of present file
                radial_file_label_list.append(present_label)

                #extract adjacent cell labels of present label using RAG
                adjacent_label=[Radial_file_extraction.index_sort(i, present_label) for i in edge_list if present_label in i]
                
                #if any candidate doesnt exist, break loop
                if len(adjacent_label)==0:
                    break
                
                
                ######################################
                # Centroid-based candidate selection #
                ######################################

                #center of gravity of adjacent cells
                cx_adjacent=self.cx_list[adjacent_label]
                cy_adjacent=self.cy_list[adjacent_label]

                #center of gravity of present cell
                cx_self=self.cx_list[present_label]
                cy_self=self.cy_list[present_label]

                #select appropriate cell for adjacent cell.
                #in this case, x cooridinate of centroid of adjacent cell is smaller than that of present cell
                cx_adj_ahead_ind=np.where(cx_adjacent < cx_self)[0]
                
                #if any candidate doesnt exist, break loop
                if len(cx_adj_ahead_ind)==0:
                    break
                
                #extract adjacent cell candidates selected by centroid positions
                cx_adj_ahead=cx_adjacent[cx_adj_ahead_ind]
                cy_adj_ahead=cy_adjacent[cx_adj_ahead_ind]
                adj_ahead_label=np.asarray(adjacent_label)[cx_adj_ahead_ind]
                
                
                ###################################
                # Angle-based candidate selection #
                ###################################               
                
                #angular restriction
                adj_ahead_label_angle=[]
                for i in range(len(cx_adj_ahead)):
                    diff_cx=cx_self-cx_adj_ahead[i]
                    diff_cy=cy_self-cy_adj_ahead[i]
 
                    #this label is listed up as the candidate of adjacent cell
                    if -1*angle_thres < np.arctan2(diff_cy, diff_cx)*180/np.pi < angle_thres:
                        adj_ahead_label_angle.append(adj_ahead_label[i])
                
                #if any candidate doesnt exist, break loop
                if len(adj_ahead_label_angle)==0:
                    break

                
                #######################################################
                # Bray-Curtis cost function-based candidate selection #
                #######################################################
                
                #You can customize your cost function based on anatomical parameters you have
                #The last anatomical parameter is the most important beacuse adjacent cell is selected based on
                #the cost value of that param.
                
                #cost restriction
                for i in range(len(anatomical_param_list)):
                    if i==0:
                        adj_ahead_label_angle_cost=[k for k in adj_ahead_label_angle if 
                                       Radial_file_extraction.cost_func(present_label, k, anatomical_param_list[i])< cost_list[i]]    
                    
                    else:
                        adj_ahead_label_angle_cost=[k for k in adj_ahead_label_angle_cost if 
                                           Radial_file_extraction.cost_func(present_label, k, anatomical_param_list[i])< cost_list[i]]
                    
                    #get cost values only in the final anatomical parameters
                    if i==len(anatomical_param_list)-1:
                        adj_ahead_label_angle_cost_value=[Radial_file_extraction.cost_func(present_label, k,  anatomical_param_list[i]) 
                                                      for k in adj_ahead_label_angle_cost 
                                                      if Radial_file_extraction.cost_func(present_label, k, anatomical_param_list[i])
                                                      < cost_list[i]]

                # if the list becomes empty, break loop
                if len(adj_ahead_label_angle_cost)==0:
                    break
                
                
                ##########################################
                # Select the most probable adjacnet cell #
                ##########################################
                
                #select the label whose cost of the last anatomical parameter is the lowest
                cost_sort_ind=np.argsort(adj_ahead_label_angle_cost_value)
                adj_ahead_label_angle_cost=adj_ahead_label_angle_cost[cost_sort_ind[0]]

                #adequate adjacent cell is set as the present cell and repeat this procedure until the break condtion 
                present_label=adj_ahead_label_angle_cost


                 #add one more restriction
                #if the adjacent cell is already assigned as another radial file, break while loop.
                if len(cell_assignment_list)==0:
                    pass
                else:
                    if present_label in cell_assignment_list:
                        break
                    else:
                        pass


            #subtract labels extracted as the radial file compoents from label_descend_list.
            #This is because this code hypothesizes that one label belongs to only one radial file.
            label_descend_list=[i for i in label_descend_list if not i in radial_file_label_list]

            #rearrange radial file label list based on the centroid coordinates
            radial_file_cx=self.cx_list[radial_file_label_list]
            radial_file_cx_sortind=np.argsort(radial_file_cx)
            radial_file_label_list=np.asarray(radial_file_label_list)[radial_file_cx_sortind]

            #save the radial file information
            radial_file_label_list_summary.append(radial_file_label_list)

            #print the current status
            if len(radial_file_label_list_summary)==1:
                print(str(len(radial_file_label_list_summary))+" radial file is extracted")
            elif len(radial_file_label_list_summary)%10==0:
                print(str(len(radial_file_label_list_summary))+" radial file is extracted")        
            elif len(label_descend_list)==0:
                print("Finally, "+str(len(radial_file_label_list_summary))+" radial files are extracted")

        return radial_file_label_list_summary
    


class Radial_file_annual_ring(Radial_file_extraction):
     """
    This class contains additional functions for extracting radial files with annual ring width in softwoods.
    To achieve the purpose, two results of radial file extraction obtained in different conditions are required.
    First one is obtained in moderate condition where length of each radial file tend to be annual ring width.
    Second is obtained in loose condition where each radial file doesn't get interrupted by annual ring boundary.
    
    """
    
    def __init__(self, image, radial_file_summary_1, radial_file_summary_2, cx_list, cy_list):
        """
        member
        ------
        image                : 2D image. 
                               any images with the same dimension of lable map is available.
        radial_file_summary_1: list in list
                               radial file extraction obtained in moderate condition
        radial_file_summary_2: list in list
                               radial file extraction obtained in loose condition
        cx_list              : list in list
                               list of x coordinates of cell centroids belonging to each radial file
        cy_list              : list in list
                               list of y coordinates of cell centroids belonging to each radial file
        
        """
        
        self.image=image
        self.radial_file_summary_1=radial_file_summary_1 #This corresponds to radial file map under severe cost condition
        self.radial_file_summary_2=radial_file_summary_2 #This corresponds to radial file map under loose cost condition
        self.cx_list=cx_list
        self.cy_list=cy_list
    
    
    #Bray-Curtis criterion
    @staticmethod
    def cost_func(param_1, param_2):
        """
        Functions for calculating Bray-Curtis index.
        
        Input
        -----
        param_1, param_2: float
                          target pair for calculating their similarities. 
        
        Return
        ------
        GS: float
            Bray-Curtis index of the anatomical parameter
        
        """
        GS=np.abs(param_1-param_2)/(param_1+param_2)
        return GS

    
    @staticmethod
    def flatten_list(l):
        """
        Function for flattening list in list to list
        
        Input
        ------
        l: list in list
           list in list you would like to flatten
        
        Return
        ------
        el: list
            flattened list
        
        """
        
        for el in l:
            if isinstance(el, list):
                yield from Radial_file_annual_ring.flatten_list(el)
            else:
                yield el
    
    
    #########################################
    # Geometry based boundary determination #
    #########################################
    
    
    #extract radial files with enough length
    def extract_full_radial_file(self, thres):
        """
        Functions for eliminating cell tips and misidentified radial files.
        Ideally, radial files run from one side to the other side of the image in loose condition.
        If a part of extracted radial files get interrupted by some reasons, their length in image width direction become short.
        This kind of files are eliminated by radial file threshold defined by image width (several thousand pixels in general) * thres (0-1).
        
        Input
        ------
        thres: float (0-1)
               width direction threshold. 
        
        Return
        ------
        radial_file_full_length: list in list
                                 list of radial files with enough length in width direction.
        
        """
        #set the threshold 
        px_thres=int(self.image.shape[1]*thres)
        ind_list_l=np.where(self.cx_list[:] < px_thres)
        ind_list_r=np.where(self.cx_list[:] > self.image.shape[1]-px_thres)
        
        #extract radial files stasfying above condition
        radial_file_full_length=[]
        for i in tqdm(range(len(self.radial_file_summary_2))):
            overlap_1=list(set(self.radial_file_summary_2[i]) & set(list(ind_list_l[0])))
            overlap_2=list(set(self.radial_file_summary_2[i]) & set(list(ind_list_r[0])))

            if len(overlap_1)>0 and len(overlap_2)>0:
                radial_file_full_length.append(self.radial_file_summary_2[i])
            else:
                pass
        
        return radial_file_full_length
    
    
    #extract candidate radial files with annual ring width
    def select_radial_file(self, radial_file_full_length):
        """
        Functions for extracting candidates of radial files with annual ring width.
        Above candidates are selected utilziing radial files obtained in loose condition. 
        
        Input
        ------
        radial_file_full_length: list in list
                                 list of radial files with enough length in width direction.
        
        Return
        ------
        radial_file_annual_length_candidate: list in list
                                             list of candidates of radial files with annual ring width.
        
        """
        #extracting radial files composed of radial files of full length
        radial_file_annual_length_candidate=[]
        for i in tqdm(range(len(radial_file_full_length))):
            radial_file_temp=[]
            for j in range(len(self.radial_file_summary_1)):
                overlap_file=list(set(self.radial_file_summary_1[j]) & set(radial_file_full_length[i]))
                if len(overlap_file)>0:
                    #rearrange overlap file
                    overlap_file_cx=self.cx_list[overlap_file]
                    overlap_file_cx_sortind=np.argsort(overlap_file_cx)
                    overlap_file=np.asarray(overlap_file)[overlap_file_cx_sortind]

                    radial_file_temp.append(overlap_file)
                else:
                    pass

            radial_file_temp=[radial_file_temp[i] for i in range(len(radial_file_temp)) if len(radial_file_temp[i])>1]
            #rearrange radial file temp
            radial_file_annual_length_candidate.append(radial_file_temp)

        radial_file_annual_length_candidate=np.asarray(radial_file_annual_length_candidate)
        
        return radial_file_annual_length_candidate
    
    
    #combine above two functions to one function
    def select_radial_file_candidate(self, thres):
        """
        Functions for picking up candidates of radial files with annual ring with.
        Detaild of this function is already mentioned in the above two functions.
        
        Input
        ------
        thres: float (0-1)
               width direction threshold.
        
        Return
        ------
        radial_file_annual_length_candidate: list in list
                                             list of candidates of radial files with annual ring width.
        
        """
        #utilize above two functions
        radial_file_full_length = Radial_file_annual_ring.extract_full_radial_file(self, thres)
        radial_file_annual_length_candidate = Radial_file_annual_ring.select_radial_file(self, radial_file_full_length)
        
        return radial_file_annual_length_candidate
        
    
    def extract_radial_file_anatomical_param(self, radial_file_annual_length_candidate, *args):
        """
        Functions for extracting anatomical parameters of cells located at the both ends of radial files candidates with annual ring width.
        In args, you can input lists of anatomical parameters you would like to use.
        
        Input
        ------
        radial_file_annual_length_candidate: list in list
                                             list of candidates of radial files with annual ring width.
        *args                              : list in list
                                             list of anatomical parameters.
        
        Return
        ------
        radial_file_cell_pos_list        : list in list
                                           centroids' cooridinates of cells located at the both end   
        radial_file_cell_dist_list       : list in list
                                           distance between cells located at the both end
        radial_file_cell_label_list      : list in list
                                           labels of cells located at the both end
        radial_file_anatomical_param_list: list in list
                                           anatomical parameters corresponding to your input of cells located at the both end
        
        
        """
        #boundary between early and latewood is defined by centroids positions of tracheids loactaed at boundary regions
        radial_file_cell_pos_list=[]
        radial_file_cell_dist_list=[]
        radial_file_cell_label_list=[]

        for i in tqdm(range(len(radial_file_annual_length_candidate))):
            cell_pos_temp=[]
            cell_dist_temp=[]
            radial_file_cell_label=[]

            for j in range(len(radial_file_annual_length_candidate[i])):
                cell_end=radial_file_annual_length_candidate[i][j][0]
                cell_int=radial_file_annual_length_candidate[i][j][-1]

                #extract coordinates of centroids
                cell_int_cx=self.cx_list[cell_int]
                cell_int_cy=self.cy_list[cell_int]
                cell_end_cx=self.cx_list[cell_end]
                cell_end_cy=self.cy_list[cell_end]
                cell_pos_temp.append([cell_int_cx, cell_int_cy])
                
                #save labels
                radial_file_cell_label.append(cell_int)
                
                #calculate distance between cells
                cell_dist=np.sqrt(np.power(cell_int_cx-cell_end_cx, 2)+np.power(cell_int_cy-cell_end_cy, 2))
                cell_dist_temp.append(cell_dist)
            
            radial_file_cell_pos_list.append(cell_pos_temp)
            radial_file_cell_dist_list.append(cell_dist_temp)
            radial_file_cell_label_list.append(radial_file_cell_label)
            
        #flatten the list
        radial_file_cell_pos_list=np.asarray(list(itertools.chain.from_iterable(radial_file_cell_pos_list)))
        radial_file_cell_dist_list=np.asarray(list(Radial_file_annual_ring.flatten_list(radial_file_cell_dist_list)))
        radial_file_cell_label_list=np.asarray(list(Radial_file_annual_ring.flatten_list(radial_file_cell_label_list)))

        
        #if another anatomical parameters are set as input, below codes will run
        #anatomical parameter list is assumed as the input of *args
        if len(args)==0:
            return radial_file_cell_pos_list, radial_file_cell_dist_list, radial_file_cell_label_list
        
        else:
            radial_file_anatomical_param_list=[]
            
            for i in range(len(args)):
                radial_file_anatomical_param_temp=[]
                for j in tqdm(range(len(radial_file_annual_length_candidate))):
                    radial_file_anatomical_param_temp_=[]
                
                    for k in range(len(radial_file_annual_length_candidate[j])):
                        cell_end=radial_file_annual_length_candidate[j][k][0]
                        cell_int=radial_file_annual_length_candidate[j][k][-1]

                        #extract coordinates of centroids
                        cell_int_cx=self.cx_list[cell_int]
                        cell_int_cy=self.cy_list[cell_int]
                    
                        #store anatomical param
                        cell_int_param=args[i][cell_int]
                        radial_file_anatomical_param_temp_.append(cell_int_param)
                    
                    #save temporary result
                    radial_file_anatomical_param_temp.append(radial_file_anatomical_param_temp_)
                    
                    #flatten the list
                    radial_file_anatomical_param_temp=list(Radial_file_annual_ring.flatten_list(radial_file_anatomical_param_temp))
                
                #save flattened list
                radial_file_anatomical_param_list.append(radial_file_anatomical_param_temp)
                
                #
                radial_file_anatomical_param_list=np.asarray(radial_file_anatomical_param_list)
            
            return radial_file_cell_pos_list, radial_file_cell_dist_list, radial_file_cell_label_list, radial_file_anatomical_param_list
        
            
    
    def extract_boundary_cell(self, radial_file_cell_pos_list, radial_file_cell_dist_list, 
                              radial_file_cell_label_list, dist_thres, angle_thres, *args):
        """
        Functions for extracting candidates of cell arrays located at annual ring boundary from cell pairs at the side of each radial files.
        Cell arrays along with annual ring boundary are extracted based on their position, distance and angle between adjacent cells.
        In addition, you can use anatomical parameters for this proedure as option.
        
        Input
        ------
        radial_file_cell_pos_list  : list in list
                                     centroids' cooridinates of cells located at the both end   
        radial_file_cell_dist_list : list in list
                                     distance between cells located at the both end
        radial_file_cell_label_list: list in list
                                     labels of cells located at the both end
        dist_lim                   : float (0-1)
                                     distance threshold for extracting cell arrays ranging from 0 to 1.
        angle_lim                  : float (0-90)
                                     angle threshold for extracting cell arrays ranging from 0-90.
        *args                      : list in list
                                     anatomical parameters corresponding to your input of cells located at the both end
        
        Return
        ------
        annual_ring_boundary_list: list in list
                                   list of cell pairs located at annual ring boundaries in each radial file.
        
        """
        
        #calculate all ditsances among centroids
        all_diffs = np.expand_dims(radial_file_cell_pos_list, axis=1) - np.expand_dims(radial_file_cell_pos_list, axis=0)
        centroid_distance = np.sqrt(np.sum(all_diffs ** 2, axis=-1))
        
        #args[0]:radial_file_anatomical_param_list
        #args[1]:anatomical_param_cost
        
        #set empty list
        annual_ring_boundary_list=[]

        #set the start point
        radial_file_cx_list=[radial_file_cell_pos_list[i][0] for i in range(len(radial_file_cell_pos_list))]
        radial_file_cy_list=[radial_file_cell_pos_list[i][1] for i in range(len(radial_file_cell_pos_list))]

        radial_file_cx_list=np.asarray(radial_file_cx_list)
        radial_file_cy_list=np.asarray(radial_file_cy_list)
        
        #set condition
        dist_lim=np.mean(radial_file_cell_dist_list)*dist_thres
        angle_lim=angle_thres
        ind_array=np.arange(0, len(radial_file_cy_list))
        
        
        while(True):
            #if ind_array is empty, break loop
            if len(ind_array)==0:
                break

            #set present cell from ind_array
            present_cell_list=radial_file_cell_label_list[ind_array]
            
            #present cell is selected from the upper part of the image
            present_cy_ind=np.argsort(radial_file_cy_list[ind_array])[0]
            present_cell_ind=ind_array[present_cy_ind]
            present_cell_label=radial_file_cell_label_list[present_cell_ind]
            
            #set empty file for saving temporary results
            annual_ring_boundary=[]
            present_cell_ind_list=[]
            
            
            while(True):
                #save present cell info
                annual_ring_boundary.append(present_cell_label)
                present_cell_ind_list.append(present_cell_ind)

                #define cx, cy and area of present cell
                present_cell_cy=radial_file_cy_list[present_cell_ind]
                present_cell_cx=radial_file_cx_list[present_cell_ind]
                #present_cell_area=radial_file_cell_area_list[present_cell_ind]
                #present_cell_width=radial_file_cell_width_list[present_cell_ind]

                #selecting candidate cells located at boundary
                #first criterion: y coordinate of next candidate becomes larger than that of present
                adjacent_cell_candidate_ind=[i for i in ind_array if radial_file_cy_list[i] > present_cell_cy]
                adjacent_cell_candidate_ind=[i for i in adjacent_cell_candidate_ind if i in ind_array]
                
                #if no candidate is found out, break loop
                if len(adjacent_cell_candidate_ind)==0:
                    break
                
                #sorting candidates by centroids distance
                cell_centroid_dist=centroid_distance[present_cell_ind, adjacent_cell_candidate_ind]
                cell_centroid_dist_sortind=np.argsort(cell_centroid_dist)
                cell_centroid_dist=np.asarray(cell_centroid_dist)[cell_centroid_dist_sortind]
                adjacent_cell_candidate_ind=np.asarray(adjacent_cell_candidate_ind)[cell_centroid_dist_sortind]

                #second criterion: centroid distance become smaller than annual ring width
                adjacent_cell_candidate_ind=[adjacent_cell_candidate_ind[i] for i in range(len(adjacent_cell_candidate_ind)) 
                                             if cell_centroid_dist[i]<dist_lim]
                
                #if no candidate is found, break loop
                if len(adjacent_cell_candidate_ind)==0:
                    break
                
                
                #if you would like to set the cost bassed on anatomical parameter, below part will run 
                if len(args)==2:
                    radial_file_anatomical_param_list=args[0]
                    anatomical_param_cost=args[1]
                    
                    for i in range(len(radial_file_anatomical_param_list)):
                        present_cell_anatomy=radial_file_anatomical_param_list[i][present_cell_ind]
                        
                        #Option criterion: features of adjacent cells are similar each other 
                        adjacent_cell_candidate_ind=[adjacent_cell_candidate_ind[j] for j in range(len(adjacent_cell_candidate_ind)) 
                                                     if Radial_file_annual_ring.cost_func(radial_file_anatomical_param_list[i]                    
                                                        [adjacent_cell_candidate_ind[j]], present_cell_anatomy) < anatomical_param_cost[i]]

                        if len(adjacent_cell_candidate_ind)==0:
                            break
                
                else:
                    pass
                

                #selection based on angle
                adjacent_cell_candidate_ind_angle=[]
                for i in range(len(adjacent_cell_candidate_ind)):
                    diff_x=radial_file_cx_list[adjacent_cell_candidate_ind[i]]-present_cell_cx
                    diff_y=radial_file_cy_list[adjacent_cell_candidate_ind[i]]-present_cell_cy

                    # if the angle of difference of two vectors are within -20<theta<20, 
                    #this label is listed up as the candidate of adjacent cell
                    if 90-1*angle_lim < np.arctan2(diff_y, diff_x)*180/np.pi < 90+angle_lim:
                        adjacent_cell_candidate_ind_angle.append(adjacent_cell_candidate_ind[i])
                
                #if no candidate is found, break loop
                if len(adjacent_cell_candidate_ind_angle)==0:
                    break

                #above list is aolready sorted by cell-cell distance
                #the nearest cell is set to adjacent cell
                adjacent_cell_ind=adjacent_cell_candidate_ind_angle[0]
                
                #update present cell info
                present_cell_label=radial_file_cell_label_list[adjacent_cell_ind]
                present_cell_ind=adjacent_cell_ind


                #subtarct adjacenct cell from ind_array 
                ind_array=[i for i in ind_array if not i == adjacent_cell_ind]
                
                #if ind_array become empty (all cells are assigned to any group), break loop
                if len(ind_array)==0:
                    break
            
            #save result
            annual_ring_boundary_list.append(annual_ring_boundary)
            
            #update ind_array
            ind_array=[i for i in ind_array if not i in present_cell_ind_list]

        return annual_ring_boundary_list
        
    
    
    def select_radial_file_by_height(self, annual_ring_boundary_list, height_thres):
        """
        Functions for selecting candidates of cell arrays located at annual ring boundary.
        This function hypothesize that y-direction (image height direction) distance between the uppermost and lowermost cells become close to the 
        height of image.
        This can be controlled by height_thres ranging from 0 to 1.
        
        Input
        ------
        annual_ring_boundary_list: list in list
                                   list of cell pairs located at annual ring boundaries in each radial file.
        height_thres             : float (0-1)
                                   threshold in height direction ranging from 0 to 1.
        
        Return
        ------
        annual_ring_boundary_list: list in list
                                   list of cell pairs located at annual ring boundaries in each radial file screened by 
                                   height direction thresolding.
        
        """
        #omit inadequate components from annual ring boundary list
        annual_ring_boundary_list=[annual_ring_boundary_list[i] for i in range(len(annual_ring_boundary_list))
                                  if len(annual_ring_boundary_list[i]) > 1]
       
        #criterion: if cells at annual ring boundary are successfully extracted, y-direction distance between the uppermost and 
        #lowermost cells become close to the height of image
        
        height=self.image.shape[0]
        annual_ring_boundary_list=[annual_ring_boundary_list[i] for i in range(len(annual_ring_boundary_list))
            if self.cy_list[annual_ring_boundary_list[i][-1]]-self.cy_list[annual_ring_boundary_list[i][0]]> height*height_thres]
        
        return annual_ring_boundary_list
    
    
    
    def pred_boundary(self, annual_ring_boundary_list, dist_thres):
        """
        Functions for returning radial files with annual ring width.
        
        Input
        ------
        annual_ring_boundary_list: list in list
                                   list of cell pairs located at annual ring boundaries in each radial file.
        dist_thres               : float (0-1)
                                   torelance to deviation of radial files from the true annual ring width.
                                   If this value is small, criterion of selecting probable radial files with annual ring with become severe.
        
        Return
        ------
        radial_file_result_list: list in list
                                 selection results of radial files with annual ring width (list)
        
        """
        #draw annual ring boundary using linear interpoolation
        f_array=[]
        for i in range(len(annual_ring_boundary_list)):
            x=self.cx_list[annual_ring_boundary_list[i]]
            y=self.cy_list[annual_ring_boundary_list[i]]
            f=interpolate.interp1d(y, x, kind='linear', fill_value='extrapolate')
            f_array.append(f)
            
        
        #below codes extract adequate radial files
        radial_file_cx=[[self.cx_list[self.radial_file_summary_1[i][0]], self.cx_list[self.radial_file_summary_1[i][-1]]] 
                          for i in range(len(self.radial_file_summary_1))]
        radial_file_cy=[[self.cy_list[self.radial_file_summary_1[i][0]], self.cy_list[self.radial_file_summary_1[i][-1]]] 
                          for i in range(len(self.radial_file_summary_1))]

        #extract adequate radial file with annual ring width
        radial_file_result_list=[]
        for i in range(len(f_array)-1):
            
            #create empty list
            pred_cx_list_late=[]
            pred_cx_list_early=[]
            
            #predict earlywood cells located at annual ring boundary
            for j in tqdm(range(len(radial_file_cy))):
                cx_late_pred_int=f_array[i+1](radial_file_cy[j][0])
                cx_early_pred_end=f_array[i](radial_file_cy[j][1])
                pred_cx_list_late.append(cx_late_pred_int)
                pred_cx_list_early.append(cx_early_pred_end)

            pred_cx_list_late=np.asarray(pred_cx_list_late)
            pred_cx_list_early=np.asarray(pred_cx_list_early)
            
            
            #pruning by latewood position
            #assumption 1: previous latewood is placed behind earlywood in next year 
            radial_file=[self.radial_file_summary_1[i] for i in range(len(self.radial_file_summary_1))
                          if pred_cx_list_early[i] > self.cx_list[self.radial_file_summary_1[i][0]] >  pred_cx_list_late[i]]

            radial_file_ind_1=[i for i in range(len(self.radial_file_summary_1))
                          if pred_cx_list_early[i] > self.cx_list[self.radial_file_summary_1[i][0]] >  pred_cx_list_late[i]]

            #assumption 2: adequate radial file has its adequate radial file length describing one year
            #set distance between centroids of cells located at ecah end of radial file
            pred_cx_list_late=pred_cx_list_late[radial_file_ind_1]
            pred_cx_list_early=pred_cx_list_early[radial_file_ind_1]
            pred_cy_list_late=[radial_file_cy[i][0] for i in radial_file_ind_1]
            pred_cy_list_early=[radial_file_cy[i][1] for i in radial_file_ind_1]

            dist_list=[np.sqrt(np.power(self.cx_list[radial_file[i][0]]-self.cx_list[radial_file[i][-1]], 2)+
                               np.power(self.cy_list[radial_file[i][0]]-self.cy_list[radial_file[i][-1]], 2)) 
                               for i in range(len(radial_file))]
            dist_pred_list=[np.sqrt(np.power(pred_cx_list_late[i]-pred_cx_list_early[i], 2)+
                               np.power(pred_cy_list_late[i]-pred_cy_list_early[i], 2)) for i in range(len(pred_cx_list_late))]

            radial_file_ind_2=[i for i in range(len(radial_file)) 
                               if dist_pred_list[i]*(1-dist_thres) < dist_list[i] < dist_pred_list[i]*(1+dist_thres)]
            
            
            #extract radial files which satisfies above two assumptions
            radial_file_result=[self.radial_file_summary_1[i] for i in range(len(self.radial_file_summary_1)) if i in radial_file_ind_1]
            radial_file_result=[radial_file_result[i] for i in range(len(radial_file_result)) if i in radial_file_ind_2]
            
            #save result
            radial_file_result_list.append(radial_file_result)
            
    
        return radial_file_result_list
