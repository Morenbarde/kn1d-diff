import numpy as np
from numpy.typing import NDArray
from scipy import interpolate

from .utils import get_local_directory, bs2dr

class Johnson_Hinnov():
    '''
    This class handles various operations involving Johnson-Hinnov rate equations and coefficients 
    [L.C.Johnson and E. Hinnov, Journal of Quantitative Spectroscopy & Radiative Transfer. vol. 13 pp.333-358]

    Attributes
    ----------
        dknot : ndarray
            B-spline knot locations in log electron density (m^-3).
        tknot : ndarray
            B-spline knot locations in log electron temperature (eV).
        order : int
            Order of the tensor-product B-spline basis (4 = cubic).
        logr_bscoef : ndarray
            B-spline coefficients for log Johnson-Hinnov population correction
            factors R(ne, Te) for neutral and ionized hydrogen, levels n = 2-6.
        logs_bscoef : ndarray
            B-spline coefficients for log effective ionization rate coefficient
            S(ne, Te) (m^3 s^-1).
        logalpha_bscoef : ndarray
            B-spline coefficients for log effective recombination coefficient
            alpha(ne, Te) (m^3 s^-1).
        a_lyman : ndarray
            Einstein A coefficients (s^-1) for H Lyman transitions (n = 2-1...16-1).
        a_balmer : ndarray
            Einstein A coefficients [s^-1] for H Balmer transitions (n = 3-2...17-2).
    '''

    def __init__(self):

        path = get_local_directory(__file__)
        jh_data = np.load(path+"/jh_bscoef.npz")

        self.dknot = jh_data['dknot']
        self.tknot = jh_data['tknot']
        self.order = jh_data['order']
        self.logr_bscoef = jh_data['logr_bscoef']
        self.logs_bscoef = jh_data['logs_bscoef']
        self.logalpha_bscoef = jh_data['logalpha_bscoef']
        self.a_lyman = jh_data['a_lyman']
        self.a_balmer = jh_data['a_balmer']



    def jhr_coef(self, Density : NDArray, Te : NDArray, Ion : int, p : int, no_null = 0):
        '''
        Evaluates the r0(p) and r1(p) coefficients from Johnson-Hinnov tables 1 or 2.
        Gwendolyn Galleher 

        Input:
            Density	- fltarr, electron density (=hydrogen ion density) (m^-3)
            Te	- fltarr, electron temperature (eV
            Ion	- integer, =0: return "recombination" coeffcient, r0(p)
                =1: return "ionization" coeffcient, r1(p)
            p	- integer, hydrogen energy level, p=1 is ground state
        Key Words: 
            create	- if set, then create bi-cubic spline coefficients for
                interpolation of r0(p) r1(p) and save them in the
                default save set. 
            No_Null	- if set, then rather than generate a NULL value when Density and Te
                        are outside the data range, compute the rate based on the min or max
                data range values.
        '''
        
        # Evaluates R coefficients 
        if np.size(Density) != np.size(Te):
            raise Exception('Number of elements of Density and Te are different!')
        if np.size(Ion) != 1:
            raise Exception('"Ion" must be a scalar')
        if np.size(p) != 1:
            raise Exception('"p" must be a scalar')
        if p < 2 or p > 6:
            raise Exception('"p" must in range 1 < p < 7')
        if Ion < 0 or Ion > 1:
            raise Exception('"Ion" must 0 or 1')
        result = np.full(Density.shape, 1.0e32)
        LDensity = np.log(Density)
        LTe = np.log(Te)
        if no_null:
            LDensity = np.maximum( LDensity, min(self.dknot))
            LDensity = np.minimum( LDensity, max(self.dknot))
            LTe = np.maximum( LTe, min(self.tknot))
            LTe = np.minimum( LTe, max(self.tknot))
            count = np.size(LDensity)
            ok = np.arange(count)
        else:
            for i in range(0, len(Density)):
                if min(self.dknot) < LDensity[i] < max(self.dknot) and min(self.tknot) < LTe[i] < max(self.tknot):
                    ok = np.append(ok, i)

        if count > 0:
            result[ok] = np.exp(bs2dr(LDensity[ok], LTe[ok], self.order, self.order, self.dknot, self.tknot, self.logr_bscoef.T[:,Ion,p-2]))



        # print("jhr_coef", result)
        # input()
        return result 
    


    def nh_saha(self, Density, Te, p):
        '''
        Evaluates the Saha equilibrium population density (m^-3)
        for atomic hydrogen 
        
        Inputs: 
            Density     - array, electron density (=hydrogen ion density) (m^-3)
            Te          - array, electron temperature (eV)
            p           - array, hydrogen energy level, p=1 is ground state
        '''

        if len(Density) != len(Te):
            raise Exception('Number of Elements of Density and Te are different!')
        if hasattr(p, "__len__"):
            raise Exception('"p" must be a scalar')
        if p<0:
            raise Exception('“p” must greater than 0')
        result = [1.0e32] * len(Density)

        ok = np.array([]) # updated how ok is defined to resolve errors - GG
        for i in range(0, len(Density)):
            if 0.0 < Density[i] < 1.0e32 and 0.0 < Te[i] < 1.e32:
                ok = np.append(ok, i)
        # converts array from a float array to an int array
        ok = ok.astype(int) 

        if len(ok) > 0:
            for i in ok:
                result[i] = Density[i] * (3.310E-28 * Density[i]) * p * p * np.exp(13.6057 / (p * p * Te[i])) / (Te[i] ** 1.5)
                # this returns many infinite values and 
                # I can't tell if that is an issue with the code inputs or if they are supposed to be that big - GG
        return result


    def lyman_alpha(self, Density : NDArray, Te : NDArray, N0 : NDArray, photons = 0, no_null = 0):
        '''
            Computes Lyman-alpha emissivity (watts m^-3) given the local
         electron density, electron temperature, and ground-state
         neutral density.
        
         Method:
            (1) Compute the local n=2 population density using the Johnson-Hinnov
                rate equations and coefficients [L.C.Johnson and E. Hinnov, J. Quant. 
                Spectrosc. Radiat. Transfer. vol. 13 pp.333-358]
            (2) Multiply by the n=2->1 spontaneous emission coefficient
            (3) Convert to watts/m^3
        
        ________________________________________________________________________________
        Input:
         	Density	- fltarr, electron density (=hydrogen ion density) (m^-3)
         	Te	- fltarr, electron temperature (eV
         	N0	- fltarr, ground state neutral density (m^-3)
        
        Keywords:
        	photons - returns emissivity in number of photons m^-3 s^-1
        	create	- if set, then create bi-cubic spline coefficients for
        		  interpolation of r0(p) r1(p) and save them in the
        		  default save set. 
        	No_Null	- if set, then rather than generate a NULL value when Density and Te
                        are not null but still outside the data range, compute the rate based on the min or max
        		  data range values.
        ________________________________________________________________________________
        History:
           Coding by B. LaBombard  6/29/99
           Coefficients from J. Terry's idl code JH_RATES.PRO
        variables in JH_coef common block - this is only temporary bc we havent finished discussing common blocks
        '''
        
        # From Johnson-Hinnov, eq (11):
        # n(2) =  ( r0(2) + r1(2) * n(1) / NHsaha(1) ) * NHsaha(2)
        if np.size(Density) != np.size(Te):
            raise Exception('Number of elements of Density and Te are different!')
        if np.size(Density) != np.size(N0):
            raise Exception(' Number of elements of Density and N0 are different! ')
        result = np.full(Density.shape,1.0e32)
        photons = np.full(Density.shape,1.0e32)
        r02 = self.jhr_coef(Density, Te, 0, 2, no_null = no_null)
        r12 = self.jhr_coef(Density, Te, 1, 2, no_null = no_null)
        NHSaha1 = self.nh_saha(Density, Te, 1)
        NHSaha2 = self.nh_saha(Density, Te, 2)
        ok=np.array([])
        for i in range(0, np.size(Density)):
            if 0 < N0[i] < 1e32 and r02[i] < 1.0e32 and r12[i] < 1.0e32 and NHSaha1[i] < 1.0e32 and NHSaha2[i] < 1.0e32:
                ok = np.append(ok, i)
        count = np.size(ok)
        if count > 0:
            for i in range(0, np.size(ok)):
                photons[i] = self.a_lyman[0] * ( r02[i] + r12[i] * N0[i] / NHSaha1[i] ) * NHSaha2[i]
                result[i] = 13.6057 * (0.75) * photons[i] * 1.6e-19
        return result


    def balmer_alpha(self, Density : NDArray, Te : NDArray, N0 : NDArray, photons = 0, no_null = 0):
        '''
        Computes Balmer-alpha emissivity (watts m^-3) given the local
        electron density, electron temperature, and ground-state
        neutral density.

        Method :
        (1) Compute the local n=3 population density using the Johnson-Hinnov
            rate equations and coefficients [L.C.Johnson and E. Hinnov, J. Quant. 
            Spectrosc. Radiat. Transfer. vol. 13 pp.333-358]
        (2) Multiply by the n=3->2 spontaneous emission coefficient
        (3) Convert to watts/m^3
        
        ________________________________________________________________________________
        Input:
            Density	- fltarr, electron density (=hydrogen ion density) (m^-3)
            Te	- fltarr, electron temperature (eV
            N0	- fltarr, ground state neutral density (m^-3)
        
        Keywords:
            photons - returns emissivity in number of photons m^-3 s^-1
            create	- if set, then create bi-cubic spline coefficients for
                interpolation of r0(p) r1(p) and save them in the
                default save set. 
            No_Null	- if set, then rather than generate a NULL value when Density and Te
                        are not null but still outside the data range, compute the rate based on the min or max
                data range values.
        ________________________________________________________________________________
        History:
        Coding by B. LaBombard  6/29/99
        Coefficients from J. Terry's idl code JH_RATES.PRO
        '''

        # From Johnson-Hinnov, eq (11):
        # n(3) = ( r(0) + r1(3) * n(1) / NHsaha(1) ) * NHsaha(3)

        if np.size(Density) != np.size(Te):
            raise Exception('Number of elements of Density and Te are different!')
        if np.size(Density) != np.size(N0):
            raise Exception(' Number of elements of Density and N0 are different! ')
        result = np.full(Density.shape,1.0e32)
        photons = np.full(Density.shape,1.0e32)
        r03 = self.jhr_coef(Density, Te, 0, 3, no_null = no_null)
        r13 = self.jhr_coef(Density, Te, 1, 3, no_null = no_null)
        NHSaha1 = self.nh_saha(Density, Te, 1)
        NHSaha3 = self.nh_saha(Density, Te, 3)
        ok=np.array([])
        for i in range(0, np.size(Density)):
            if 0 < N0[i] < 1e32 and r03[i] < 1.0e32 and r13[i] < 1.0e32 and NHSaha1[i] < 1.0e32 and NHSaha3[i] < 1.0e32:
                ok = np.append(ok, i)
        count = np.size(ok)
        if count > 0:
            for i in range(0, np.size(ok)):
                photons[i] = self.a_balmer[0] * ( r03[i] + r13[i] * N0[i] / NHSaha1[i] ) * NHSaha3[i]
                result[i] = 13.6057 * ( 0.25 - 1.0 / 9.0 ) * photons[i] * 1.6e-19
        return result