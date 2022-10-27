##################################################################
##  Version 2.0.0 October 10, 2021
##  RegakuRapid2 application: 
##################################################################
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import cv2
from tqdm import tqdm
import pandas as pd
from scipy.optimize import curve_fit
import openpyxl
import datetime


in_path='./data' # 
profile_path='./results/profile'
image_path='./results/image'
figure_path='./results/figure'
peakfit_path='./results/peakfit'

legends=[['(004)','Azimuthal intensity distribution'],['2 theta','(004) Intensity (a.u.)'],\
         ['Azimuthal angle','Intensity (a.u.)'],\
         ['Equators','Azimuthal intensity distribution'],['2 theta','Equatorial Intensity (a.u.)'],\
         ['Azimuthal angle','Intensity (a.u.)']]

class RigakuRapid2(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.rows=2800 # initial IP size
        self.cols=2560 #
        self.u=0.1 # pixel resolution (u) in mm
        self.L=127.4 # camera-sample distance (L) in mm
        self.centerX=1448  # beam stop position X
        self.centerY=1272  # beam stop position Y
        self.out=np.zeros((2400,2400,3)) # array for handling 3 types of images
        # azimutal anglar range
        self.az_rg=45  # range + - direction in degree
        self.az_stp=0.15 # azimuthal angle step in degree
        # 2-theta range
        # 004 -> 32.6 34.6. 37.6  2t-heta range for 004 reflection
        # equatorial -> 10, 30 2-theta range for equatorial refrections
        self.me_min=32.6 # meridional minimum 2-theta
        self.me_max=36.6 # meridional maximum 2-theta
        self.eq_min=10.0 # equatorial minimum 2-theta
        self.eq_max=30.0 # equatorial maximum 2-theta
        # azmin[ minimum asimuthal angle of 004 and equator]
        # azmax[ maximum azimuthal angle of 004 and equator]
        # eq_std: standard azymutal angle for equator
        self.az_min=[int((90-self.az_rg)/self.az_stp),int((180-self.az_rg)/self.az_stp)]
        self.az_max=[int((self.az_rg+90)/self.az_stp),int((self.az_rg+180)/self.az_stp)]
        self.me_std=int((90)/self.az_stp)
        self.eq_std=int((180)/self.az_stp)
        # directory
        self.in_path='./data' # 
        self.profile_path='./results/profile'
        self.image_path='./results/image'
        self.figure_path='./results/figure'
        self.peakfit_path='./results/peakfit'
        #
    def create_dir(self, path):
        s_path=os.path.join(self.in_path,path)
        p_path=os.path.join(self.profile_path,path)
        i_path=os.path.join(self.image_path,path)
        f_path=os.path.join(self.figure_path,path)
        pf_path=os.path.join(self.peakfit_path,path)
        os.makedirs(s_path, exist_ok=True)
        os.makedirs(p_path, exist_ok=True)
        os.makedirs(i_path, exist_ok=True)
        os.makedirs(f_path, exist_ok=True)
        os.makedirs(pf_path, exist_ok=True)
        return s_path,p_path,i_path,f_path,pf_path
            
    def imread(self, in_path): # read raw image file
        fd = open(in_path, 'rb')
        rawdata=fd.read()
        fd.close()
        f = Image.frombytes('F', (self.rows,self.cols), rawdata, "raw",'F;16B')
        im =np.array(f)
        return im
    
    def imcrop(self, img): # trim area in 1200 X 1200
        im_t= img[(self.centerY-1200):(self.centerY+1200), (self.centerX-1200):(self.centerX+1200)]
        return im_t
    
    def flat(self, img): # convert cylindrical to cartesian camera-sample distance (L) in mm
        # pixel size (u) in mm
        rows, cols = img.shape # 0 vertical direction 1 horizontal
        center_y = int(cols / 2) # horizontal center
        center_x = int(rows / 2) # vertical center
        fp_max=center_y    
        flat = np.zeros((fp_max*2,fp_max*2))
        for  x in tqdm(range(fp_max*2)):   
            theta=math.atan((x-center_x)*self.u/self.L)
            dx, point_x =math.modf(self.L*theta/self.u+center_x)
            for y in range(fp_max*2):       
                dy, point_y =math.modf(self.L*(y-center_y)/(np.sqrt(self.L**2+((x-center_x)*self.u)**2))+ center_y)
                if point_x > cols or point_x < 0 or point_y > rows or point_y < 0:
                    pass
                else:
                    sy=(img[int(point_y)-1, int(point_x)] *(1-dy)+img[int(point_y), int(point_x)] *dy)
                    sx=(img[int(point_y), int(point_x)-1] *(1-dx)+img[int(point_y), int(point_x)] *dx)
                    flat[y, x] =(sy+sx)
        return flat
    
    
    def polar(self, img): # convert cartesian to polar coodinate
        value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
        polar_image = cv2.linearPolar(img,(img.shape[1]/2, img.shape[0]/2), value, cv2.WARP_FILL_OUTLIERS)
        polar = polar_image.astype(np.uint16)
        return polar
    
    def conv_set(self, img): # conversion to one set cylindrical, cartesian, polar image in 3 channel
        self.out[:,:,0]=self.imcrop(img)
        self.out[:,:,1]=self.flat(self.out[:,:,0])
        self.out[:,:,2]=self.polar(self.out[:,:,1])
        return self.out
    
    def profile(self, img): # panda df from profile image
        # theta=arctan(u*x/(np.sqrt(2)*L)
        num_y,num_x= img.shape
        y_res=360/img.shape[0]
        x_max=img.shape[1]
        theta_max=math.atan(self.u*x_max/(np.sqrt(2)*self.L))*180/np.pi
        x_step=[math.atan(self.u*x/(np.sqrt(2)*self.L))*180/np.pi for x in range(x_max)]  # 2 theta
        index=np.arange(0, 360, y_res)
        column_name=x_step
        df=pd.DataFrame(img,index=index, columns=column_name)
        return df  # 
        
    def me_eq(self, df):
        rad_me=[np.where(df.columns >= self.me_min)[0][0],np.where(df.columns >= self.me_max)[0][0]]
        rad_eq=[np.where(df.columns >= self.eq_min)[0][0],np.where(df.columns >= self.eq_max)[0][0]]
        peak004=np.argmax(df.iloc[self.az_min[0]:self.az_max[0],rad_me[0]:rad_me[1]].sum(axis=0))
        peak200=np.argmax(df.iloc[self.az_min[0]:self.az_max[0],rad_eq[0]:rad_eq[1]].sum(axis=0))        
        df_004_radial=df.iloc[self.me_std-50:self.me_std+50,rad_me[0]:rad_me[1]].sum(axis=0)  #radial profile
        df_004_azimuth=df.iloc[self.az_min[0]:self.az_max[0],rad_me[0]+peak004-20:rad_me[0]+peak004+20].sum(axis=1)   # azimuthal profile
        df_eq_radial=df.iloc[self.eq_std-50:self.eq_std+50,rad_eq[0]:rad_eq[1]].sum(axis=0)  # radial profile
        df_eq_azimuth=df.iloc[self.az_min[1]:self.az_max[1],rad_eq[0]+peak200-20:rad_eq[0]+peak200+20].sum(axis=1)  # azimuthal profile
        im_004=df.iloc[self.az_min[0]:self.az_max[0],rad_me[0]:rad_me[1]]
        im_eq=df.iloc[self.az_min[1]:self.az_max[1],rad_eq[0]:rad_eq[1]]       
        return im_004,df_004_radial,df_004_azimuth,im_eq,df_eq_radial,df_eq_azimuth
    
    def me_eq_adjust(self, df, MFA):
        az_MFA=np.where(df.columns >= MFA)
        rad_me=[np.where(df.columns >= self.me_min)[0][0],np.where(df.columns >= self.me_max)[0][0]]
        rad_eq=[np.where(df.columns >= self.eq_min)[0][0],np.where(df.columns >= self.eq_max)[0][0]]
        peak004=np.argmax(df.iloc[self.az_min[0]:self.az_max[0],rad_me[0]:rad_me[1]].sum(axis=0))
        peak200=np.argmax(df.iloc[self.az_min[0]:self.az_max[0],rad_eq[0]:rad_eq[1]].sum(axis=0))        
        df_004_azimuth=df.iloc[self.me_std+az_MFA-5:self.me_std+az_MFA+5,\
                               rad_me[0]+peak004-20:rad_me[0]+peak004+20].sum(axis=1)   # azimuthal profile
        df_eq_azimuth=df.iloc[self.eq_std+az_MFA-5:self.eq_std+az_MFA+5,\
                              rad_eq[0]+peak200-20:rad_eq[0]+peak200+20].sum(axis=1)  # azimuthal profile
        return df_004_radial,df_eq_radial

        





    


    

