# po_mfa_2022
Softwares developped by pywood21: 

Source: Yusuke Kita, Tatsuya Awano, Arata Yoshinaga and Junji Sugiyama: Intra-annual fluctuation in morphology and microfibril angle of tracheids revealed by novel microscopy-based imaging. Journal DOI

 



Part I: a new CV technique combining polarization optical microscopy, fluorescence microscopy, and image segmentation.

Part II: X-ray fiber diffraction analysis to measure MFA by peak deconvolution method.



# Part I

## Radial file based evaluation of softwood cell anatomy and MFA.



<img src="underconstruction.png" alt="under_construction" style="zoom:25%;" />





# Part II

## Data deconvolution of fiber diagrams obtained by Rigaku Rapid II 



1) **Read raw data  : read and convert**
   1) Original data is recoded by cylindrical IP camera. Since MFA measurement requires azimuthal intensity profile, one needs to convert cylindirical to cartesian coodinate system
   2) Coordinate system is then converted to polar coordinate system ( ***r***, ***phi*** ). ***r*** is a radial distance from the center (center_x, center_y), ***phi*** is a angle from the horizontal x axis, taken clockwizely.
   3) Original and converted data (intensity data in 2400 x 2400 size of 32bit depth) are saved in one **numpy** (**.npz** ) file in three channels. Channels 0,1,2 corresponds to cylindrical, cartesian, polar coordinate data, respectively.

2. **Line profile for MFA measurement**
   1. Polar coordinate data is reshaped in **panda** dataframe adjusting '**index**' to 'azimuthal angle', '**columns**' to '2-theta angle'. ROI in polar projected data can be obtained for both azimuthal and 2-theta range and in this default settiing, (004) and equators (11-0),(110),(200).  
   2. Intensity profiles are obtained from thin-slip from the ROI by integration in either azimuthal or radial directions. Profile data are saved in '***.pkl***' file.

3. **Deonvolution by peakfitting**
   1. 2 equivalent gaussian functions, having common amplitude and standard deviation, distanced 2*MFA are assumed and will be fit to the original profile.
   2. Results are summerized in Excel form.



**To start the program:**

Create data directory under ./data. For instance, if you have multiple dataset to be analysed, save data in 'cellulose20210909' directory and placed in under./data.

In the 01_Read... program, you rewrite

```python
my_data_set = 'cellulose20210909'
```

Then run the script. In the order of X01_Read_Convert, X02_Profile_, X03_Fitting. All the necessary files and directories will be generated sequentially.
