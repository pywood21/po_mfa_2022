"""
This .py file contains some functions for obtaining retardation values and MFA from retaradtion.
Upper part is described for calculation method proposed in Kita & Sugiyama (2021) Holzforschung [1].(Hyperspectral based method)
Middle part corresponds to PolScope-type mesurement proposed in Shribak & Oldenbourg (2003) Applied Optics [2].(PolScope method)
In the PolScope method, rotatable polarizer is used instead of liquid-crystal tunable retarder.
Therefore, chi is fixed to 90 degree, so that some restrictions are added comapred with original one. 
Lower part orresponds to functions for converting retardation to MFA (Abraham & Elbaum (2013) New Phytologist [3]).

References
----------
[1] Kita Y, Sugiyama J. 2021. Wood identification of two anatomically similar Cupressaceae species based on two-dimensional microfibril angle 
    mapping. Holzforschung 75: 591-602.
[2] Shribak M, Oldenbourg R. 2003. Techniques for fast and sensitive measurements of two-dimensional birefringence distributions. Applied Optics 
    42: 3009-3017.
[3] Abraham Y, Elbaum R. 2013. Quantification of microfibril angle in secondary cell walls at subcellular resolution by means of polarized light   
    microscope. New Phytologist 197: 1012-1019.

"""

#import libraries
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import math
from lmfit import Model
from tqdm import tqdm



##############################
# Hyperspectral based method #
##############################

"""
Ret func class is for calculating retrardation based on hyperspectral image
Originally proposed in Kita & Sugiyama (2021) Holzforschung [1].
Theoretical background and some limitations are exhaustively discussed in the master thesis of Y.K.
"""

class Ret_func:
    def __init__(self, wv, st_ind, end_ind):
        self.wv=wv #wavelength region for taking hyperspectral images
        self.st_ind=st_ind #start point for fitting to calculate retardation
        self.end_ind=end_ind #end point for fitting to calculate retardation
    
    """
    Below function is model transmission intensity curve of the certain retardation value 
    Details are well decribed in Tsuboi (1966) [4]
    
    References
    ----------
    [4] Tsuboi S. 1966. "Polarization optical microscope (in Japanese)" Tokyo, Japan: Iwanami Shoten, Publishers.
    """
    
    @staticmethod
    def Intensity(x, R, a, b):
        """Calculate intensity curve at the certain retardation value within visible light wavelength ranges.
        
        Parameters
        ----------
        x: array-like or float
            Wavelength (nm)
        R: float
            Retardation value (nm)
        a, b: float
            Scaling factors (a.u.).
        
        Returns
        -------
        intensity curve: array-like or float
            Intensity curve at the certain retardation value within visible light wavelength ranges.
     
        """
        return a*pow(np.sin(np.pi*R/x),2)+b
    
    
    def Ret_calc(self, target_array):
        """Calculate retardation from original hyperspectral intensity curves
        
        Parameters
        ----------
        target_array: (N, M) array
            Flattened hyperspectral images.
            N corresponds to pixel numbers of a single image. If original image is n*m pixels, N=n*m.
            M is equal to a point number of a wavelength axis in hyperspectral imaging.
        
        Returns
        -------
        Fitting results: list
            Returned values are list of two scaling factors, a and b, and R after fitting.
        
        """
        #lmfit library is used 
        Intense = Model(Ret_func.Intensity ,prefix='I_')
        mod=Intense
        param = Intense.make_params()
        #set fitting tolerance and intial values
        param['I_R'].set(530, min=300, max=750) #retaradtion
        param['I_a'].set(0.1 ,min=0, max=2) #scaling parameter
        param['I_b'].set(0.02 ,min=0, max=0.3) #scaling parameter
        
        list_len=len(target_array)
        
        #return fitting results
        return list(map(lambda i:mod.fit(target_array[self.st_ind:self.end_ind,i], 
                                         param, x=self.wv[self.st_ind:self.end_ind]).best_values, tqdm(range(list_len))))
        #returned results are dict objects containing the best values of I_R, I_a and I_b


"""
Below Ret_func_mp is the same function to Ret_func but multicore claculation is available in this ver,
Depending on the situation, generally multiprocessing makes our calculation several times faster compared with single core.
"""

class Ret_func_mp(Ret_func):
    def __init__(self, core_num, target_array, wv, st_ind, end_ind):
        self.core_num=core_num #core number for multiprocessing
        self.target_array=target_array #target array is set beforehand for multiprocessing
        self.wv=wv #wavelength region for taking hyperspectral images
        self.st_ind=st_ind #start point for fitting to calculate retardation
        self.end_ind=end_ind #end point for fitting to calculate retardation
    
    
    def Ret_loop(self, num):
        """Calculate retardation from from original hyperspectral intensity curves by using multiprocessing.
        
        Parameters
        ----------
        num: int
            Core number, Please see examples of how to use in PolScope_mp class.
        
        Returns
        -------
        Fitting results: list
            Returned values are list of two scaling factors, a and b, and R after fitting.
         
        """
  
        #lmfit is used
        Intense = Model(Ret_func.Intensity ,prefix='I_')
        mod=Intense
        param = Intense.make_params()
        #set fitting tolerance and intial values
        param['I_R'].set(530, min=300, max=750) #retaradtion
        param['I_a'].set(0.1 ,min=0, max=2) #scaling parameter
        param['I_b'].set(0.02 ,min=0, max=0.3) #scaling parameter

        #multiprocessing condition
        ini = self.target_array.shape[1] * num // (self.core_num)
        fin = self.target_array.shape[1] * (num+1) // (self.core_num)
        
        #return fitting results
        return list(map(lambda i:mod.fit(self.target_array[self.st_ind:self.end_ind,i], 
                                         param, x=self.wv[self.st_ind:self.end_ind]).best_values, tqdm(range(ini,fin))))
        #returned results are dict objects containing the best values of I_R, I_a and I_b
        

        
        
###################
# PolScope method #
###################        

"""
Below function is for calculating retardation and azimuthal angle by brand-new method originally proposed by Oldenbourg & Mei (1995) Journal of Microscopy [5].
Theoretical background is exhaustively explained in the aritcle by Shribak & Oldenbourg (2003) Applied Optics [2]. 
Above paper proposed a lot of algorithms for measuring retardation.
Following, only three and four-frame without extinction setting algorithms are implemented.

References
----------
[5] Oldenoburg R, Mei G. 1995. New polarized light microscope with precision universal compensator. Journal of Microscopy 180: 140-147.

"""

class Polscope_func:
    
    ##########################################
    # Three-frame without extinction setting #
    ##########################################
    
    """
    Below two functions are for calculating term A and B by using "three frame without extinction setting" originally proposed in Shribak &   
    Oldenbourg (2003) Applied Optics [2].
    setting1, 2, 3 correspond to (alpha=0, beta=180), (alpha=180, beta=180) and (alpha=90, beta=90).
    => Corresponding polarizer angles are 0, 90 and 45 degree.
    """
    
    @staticmethod
    def term_A_three(setting_1, setting_2, setting_3, bg, bg_sub="True"):
        """Calculate term A in "three frame without extinction setting" algorithm.
        
        Parameters
        ----------
        setting_1, setting_2, setting_3: 1d array
            Flattened image in each setting (polarizer angle).
        bg: 1d array
            Flattened image without any illumination (containig only intrinsic noises of a CCD camera)
        bg_sub: "True" or else
            If bg_sub=="True", subtract intrinsic noises of a CCD from the images of each setting.
        
        Returns
        -------
        term_A: 1d array
            Calculated term A from each setting.
        
        """
        
        if bg_sub=="True":
            term_A=(setting_1-setting_3)/(setting_1+setting_2-2*bg)
        else:
            term_A=(setting_1-setting_3)/(setting_1+setting_2)
        
        return term_A
    
    
    @staticmethod
    def term_B_three(setting_1, setting_2, setting_3, bg, bg_sub="True"):
        """Calculate term B in "three frame without extinction setting" algorithm.
        
        Parameters
        ----------
        setting_1, setting_2, setting_3: 1d array
            Flattened image in each setting (polarizer angle).
        bg: 1d array
            Flattened image without any illumination (containig only intrinsic noises of a CCD camera)
        bg_sub: "True" or else
            If bg_sub=="True", subtract intrinsic noises of a CCD from the images of each setting.
        
        Returns
        -------
        term_A: 1d array
            Calculated term B from each setting.
        
        """
        
        if bg_sub=="True":
            term_B=(setting_2-setting_3)/(setting_1+setting_2-2*bg)
        else:
            term_B=(setting_2-setting_3)/(setting_1+setting_2)
            
        return term_B
    
    
    """
    Below two functions are for calculating retardation and azimuthal angle from termA and B.
    """
    
    
    @staticmethod
    def calc_delta_three(A, B):
        """Calculate retardation (delta) from term A and term B obtained from "three frame without extinction setting" algorithm.
        
        Parameters
        ----------
        A, B: 1d array
            Calculated term A and B by above the functions.
        
        Returns
        -------
        delta: 1d array
            Retardation calculated from term A and B (radian).
        
        """
        
        delta=2*np.arctan(((2*(A**2+B**2))**0.5)/(1+((1-2*(A**2+B**2))**0.5)))
        return delta
    
    
    @staticmethod
    def calc_phi_three(A, B):
        """Calculate azimuthal angle (phi) from term A and term B obtained from "three frame without extinction setting" algorithm.
        
        Parameters
        ----------
        A, B: 1d array
            Calculated term A and B by above the functions.
        
        Returns
        -------
        phi: 1d array
            Azimuthal angle calculated from term A and B (radian).
        
        """
        
        if B==0:
            if A>=0:
                return np.pi/8
            if A<0:
                return np.pi*5/8

        else:
            phi=0.5*np.arctan(A/B)-np.pi/8
            if (A>=0)&(B>=0):
                if phi>=0:
                    return phi
                if phi<0:
                    return phi+np.pi

            if (A>=0)&(B<0):
                return phi+np.pi/2

            if (A<0)&(B<0):
                return phi+np.pi/2

            if (A<0)&(B>=0):
                return phi+np.pi

            
          
    #########################################
    # Four-frame without extinction setting #
    #########################################
    
    """
    Below two functions are for calculating term A and B by using "three four without extinction setting" originally proposed in Shribak &   
    Oldenbourg (2003) Applied Optics [2].
    setting1, 2, 3, 4 correspond to (alpha=0, beta=180), (alpha=180, beta=180) and (alpha=90, beta=90), (alpha=90, beta=270) .
    => Corresponding polarizer angles are 0, 90, 45, 135 degree.
    """
    
    #termA
    @staticmethod
    def term_A_four(setting_1, setting_2, bg, bg_sub="True"):
        """Calculate term A in "four frame without extinction setting" algorithm.
        
        Parameters
        ----------
        setting_1, setting_2: 1d array
            Flattened image in each setting (polarizer angle).
        bg: 1d array
            Flattened image without any illumination (containig only intrinsic noises of a CCD camera)
        bg_sub: "True" or else
            If bg_sub=="True", subtract intrinsic noises of a CCD from the images of each setting.
        
        Returns
        -------
        term_A: 1d array
            Calculated term A from each setting.
        
        """
        
        if bg_sub=="True":
            term_A=(setting_1-setting_2)/(setting_1+setting_2-2*bg)
        else:
            term_A=(setting_1-setting_2)/(setting_1+setting_2)
        
        return term_A
    
    
    @staticmethod
    def term_B_four(setting_3, setting_4, bg, bg_sub="True"):
        """Calculate term B in "four frame without extinction setting" algorithm.
        
        Parameters
        ----------
        setting_3, setting_4: 1d array
            Flattened image in each setting (polarizer angle).
        bg: 1d array
            Flattened image without any illumination (containig only intrinsic noises of a CCD camera)
        bg_sub: "True" or else
            If bg_sub=="True", subtract intrinsic noises of a CCD from the images of each setting.
        
        Returns
        -------
        term_A: 1d array
            Calculated term B from each setting.
        
        """
        
        if bg_sub=="True":
            term_B=(setting_4-setting_3)/(setting_4+setting_3-2*bg)
        else:
            term_B=(setting_4-setting_3)/(setting_4+setting_3)
        
        return term_B
    
    
    #Below two functions are for calculating retardation and azimuthal angle from termA and B.
    #Inputs of both two funtions are A and B calculating above.
    
    #Retardation
    @staticmethod
    def calc_delta_four(A, B):
        """Calculate retardation (delta) from term A and term B obtained from "four frame without extinction setting" algorithm.
        
        Parameters
        ----------
        A, B: 1d array
            Calculated term A and B by above the functions.
        
        Returns
        -------
        delta: 1d array
            Retardation calculated from term A and B (radian).
        
        """
        
        delta=2*np.arctan(((A**2+B**2)**0.5)/(1+(1-(A**2+B**2))**0.5))
        return delta

    #Azimuthal angle
    @staticmethod
    def calc_phi_four(A, B):
        """Calculate azimuthal angle (phi) from term A and term B obtained from "four frame without extinction setting" algorithm.
        
        Parameters
        ----------
        A, B: 1d array
            Calculated term A and B by above the functions.
        
        Returns
        -------
        delta: 1d array
            Azimuthal angle calculated from term A and B (radian).
        
        """
        
        if B==0:
            if A>=0:
                return np.pi/4
            if A<0:
                return np.pi*3/4

        else:
            phi=0.5*np.arctan(A/B)
            if (A>=0)&(B>=0):
                return phi

            if (A>=0)&(B<0):
                return phi+np.pi/2

            if (A<0)&(B<0):
                return phi+np.pi/2

            if (A<0)&(B>=0):
                return phi+np.pi

"""
Below functions are supported by multicore processing when retardation and azimuthal angle are calculated.
"""

class Polscope_mp(Polscope_func):
    def __init__(self, core_num, term_A, term_B):
        self.core_num=core_num #core number for multiprocessinf
        self.term_A=term_A #termA is set beforehand
        self.term_B=term_B #termB is set beforehand
    
    
    ##########################
    # Three-frame algorithm  #
    ##########################
    
    def calc_delta_three_mp(self, num):
        """Calculate retardation from term A and B in "three frame algorithm without extinction setting" by using multiprocessing.
        
        Parameters
        ----------
        num: int
            Core number, Please see examples of how to use in PolScope_mp class.
        
        Returns
        -------
        Fitting results: list
            Returned values are list of retardations (radian).
            
        Examples
        --------
        #set core number
        >>> core_num=3
        
        #call class
        #term A and B must be calculated beforehand.
        >>> polscope_mp=Ret_MFA_func.Polscope_mp(core_num=core_num, term_A=A_true, term_B=B_true)

        #set core number and perform multiprocessing like below.
        >>> with Pool(core_num) as p:
                callback_ret = p.map(polscope_mp.calc_delta_three_mp, range(core_num)) #retardation (radian)
                callback_phi = p.map(polscope_mp.calc_phi_three_mp, range(core_num)) #azimuthal angle (radian)

        #rearrange returned lists.
        #In the above case, returned list contains 3 lists returned from 3 cores.
        >>> delta_array=[]
        >>> phi_array=[]
        >>> for i in range(len(callback_ret)):
                delta_array.extend(callback_ret[i])
                phi_array.extend(callback_phi[i])
        
        #convert list to array
        >>> delta_array=np.asarray(delta_array)
        >>> phi_array=np.asarray(phi_array)
        
        """
        
        ini = len(self.term_A) * num // (self.core_num)
        fin = len(self.term_A) * (num+1) // (self.core_num)

        return list(map(lambda i: Polscope_func.calc_delta_three(self.term_A[i], self.term_B[i]), tqdm(range(ini,fin))))
    
    
    def calc_phi_three_mp(self, num):
        """Calculate azimuthal angle from term A and B in "three frame algorithm without extinction setting" by using multiprocessing.
        
        Parameters
        ----------
        num: int
            Core number, Please see examples of how to use in PolScope_mp class.
        
        Returns
        -------
        Fitting results: list
            Returned values are list of azimuthal angles (radian).
         
        """
        
        #multiprocessing condition
        ini = len(self.term_A) * num // (self.core_num)
        fin = len(self.term_A) * (num+1) // (self.core_num)

        return list(map(lambda i: Polscope_func.calc_phi_three(self.term_A[i], self.term_B[i]), tqdm(range(ini,fin))))
    
    
    #########################
    # Four-frame algorithm  #
    #########################
    
    
    def calc_delta_four_mp(self, num):
        """Calculate retardation from term A and B in "four frame algorithm without extinction setting" by using multiprocessing.
        
        Parameters
        ----------
        num: int
            Core number, Please see examples of how to use in PolScope_mp class.
        
        Returns
        -------
        Fitting results: list
            Returned values are list of retardations (radian).
         
        """
        
        #multiprocessing condition
        ini = len(self.term_A) * num // (self.core_num)
        fin = len(self.term_A) * (num+1) // (self.core_num)

        return list(map(lambda i: Polscope_func.calc_delta_four(self.term_A[i], self.term_B[i]), tqdm(range(ini,fin))))
    
    
    def calc_phi_four_mp(self, num):
        """Calculate azimuthal angle from term A and B in "four frame algorithm without extinction setting" by using multiprocessing.
        
        Parameters
        ----------
        num: int
            Core number, Please see examples of how to use in PolScope_mp class.
        
        Returns
        -------
        Fitting results: list
            Returned values are list of azimuthal angles (radian).
         
        """
        #multiprocessing condition
        ini = len(self.term_A) * num // (self.core_num)
        fin = len(self.term_A) * (num+1) // (self.core_num)

        return list(map(lambda i: Polscope_func.calc_phi_four(self.term_A[i], self.term_B[i]), tqdm(range(ini,fin))))

    
    
######################################
# Conversion from retardation to MFA #
######################################

"""
Theory are well described in Abraham & Elbaum (2013) New Phytoloist.
Multiprocessing is available in below all functions 
"""

class MFA_func_mp:
    def __init__(self, core_num, target_array, d):
        self.core_num=core_num #core number for multiprocessing
        self.target_array=target_array #retardation values 1d array (please flatten your image to 1D)
        self.d=d #net thikness of your transverse section (nm). In our work, d=0.5*section thickness. 
        #0.5 means cellulose ratio in S2 wall of tracheid (approximately 50%, see Panshin & De Zeeuw (1980)).
    
    """
    Below four functions are both for converting retardation to MFA.
    The former two are for hyperspetral based method, the latter two are for PolScope method.
    Please select the optimal one.
    
    
    """
    
    ### Hyperspectral based method ###
    
    @staticmethod
    def ret_MFA(R, d, no=1.529, ne=1.599): #R: retardation (nm)
        """Calculate MFA from retardation. This function is for Hyperspectral based method.
        
        Paramters
        ---------
        R: 1d array or float
            Retardations (nm)
        d: float
            Net thickness of a transverse section (nm).
            It can be calculated by that d=r*t. (r: cellulose ration in cell wall, t: transverse section thickness (nm)).
            Cellulose ratio of S2 layer in softwood is near to 50% (Panshin & DeZeeuw (1980) [6]).
        no, ne: float (default: no=1.529, ne=1.599)
            Refractive indices of ordinary and extraordinary lights of cellulose crystal.
            Refractive indices exerpt from Iyre et al. (1968) Journal of Polymer Science Part A-2. Polymer Physics [7] are set as default. 
        
        Retuens
        -------
        theta: float
            MFA (degree)
        
        References
        ----------
        [6] Panshin AJ, De Zeeuw C. 1980. Textbook of wood technology. Structure, identification, properties, and uses of the commercial woods of 
            the United States and Canada. 4th edition. New York, USA: McGraw-Hill. 
        [7] Iyer KRK, Neelakantan P, Radhakrishnan T. 1968. Birefringence of native cellulose fibers. I. Untreated cotton and ramie. Journal of 
            Polymer Science Part A-2. Polymer Physics 6: 1747-1758.
            
        """
        
        R_cell=np.abs(R-530) #subtract 530 nm coming from a sensitive color plate
        radian=math.acos(np.sqrt((no**2/(ne**2-no**2))*(ne**2/(math.pow(R_cell/d+no, 2))-1))) #returned values are in radian unit
        theta=radian*180/np.pi #convert radian to degeree
        return theta
    
    
    def MFA_loop(self, num):
        """Calculate MFA from retardation by using multiprocessing. This function is for Hyperspectral based method.
        
        Parameters
        ----------
        num: int
            Core number, Please see examples of how to use in PolScope_mp class.
        
        Returns
        -------
        Fitting results: list
            Returned values are list of azimuthal angles (radian).
         
        """
        
        #multiprocessing condition
        ini = len(self.target_array) * num // (self.core_num)
        fin = len(self.target_array) * (num+1) // (self.core_num)

        return list(map(lambda i: MFA_func_mp.ret_MFA(self.target_array[i] , self.d), tqdm(range(ini,fin))))
    
    
    ### PolSope method ###
    
    @staticmethod 
    def ret_MFA_polscope(R, d, no=1.529, ne=1.599): #for multiprocessing
        """Calculate MFA from retardation. This function is for PolScope method.
        
        **Caution**
        PolScope method return retardation values in radian unit.
        Please convert radian to nm before MFA conversion step.
        
        
        Paramters
        ---------
        R: 1d array or float
            Retardations (nm)
        d: float
            Net thickness of a transverse section (nm).
            It can be calculated by that d=r*t. (r: cellulose ration in cell wall, t: transverse section thickness (nm)).
            Cellulose ratio of S2 layer in softwood is near to 50% (Panshin & DeZeeuw (1980) [6]).
        no, ne: float (default: no=1.529, ne=1.599)
            Refractive indices of ordinary and extraordinary lights of cellulose crystal.
            Refractive indices exerpt from Iyre et al. (1968) Journal of Polymer Science Part A-2. Polymer Physics [7] are set as default. 
        
        Returns
        -------
        theta: float
            MFA (degree)
        
        """
        
        radian=math.acos(np.sqrt((no**2/(ne**2-no**2))*(ne**2/(math.pow(R/d+no, 2))-1)))  #returned values are in radian unit
        theta=radian*180/np.pi #convert radian to degeree
        return theta
    
    
    #for multiprocessing
    def MFA_loop_polscope(self, num):
        """Calculate MFA from retardation by using multiprocessing. This function is for PolScope method.
        
        Parameters
        ----------
        num: int
            Core number, Please see examples of how to use in PolScope_mp class.
        
        Returns
        -------
        Fitting results: list
            Returned values are list of MFA (degree).
         
        """
        
        #multiprocessing condition
        ini = len(self.target_array) * num // (self.core_num)
        fin = len(self.target_array) * (num+1) // (self.core_num)

        return list(map(lambda i: MFA_func_mp.ret_MFA_polscope(self.target_array[i] , self.d), tqdm(range(ini,fin))))

