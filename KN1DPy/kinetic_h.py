from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import copy

from .utils import sval, get_config
from .make_dvr_dvx import VSpace_Differentials
from .create_shifted_maxwellian import create_shifted_maxwellian
from .kinetic_mesh import KineticMesh

from .sigma.collrad_sigmav_ion_h0 import collrad_sigmav_ion_h0
from .jh_related.jhs_coef import jhs_coef
from .sigma.sigmav_ion_h0 import sigmav_ion_h0
from .jh_related.jhalpha_coef import jhalpha_coef
from .sigma.sigmav_rec_h1s import sigmav_rec_h1s
from .sigma.sigma_cx_h0 import sigma_cx_h0
from .sigma.sigma_el_h_h import sigma_el_h_h
from .sigma.sigma_el_h_hh import sigma_el_h_hh
from .sigma.sigma_el_p_h import sigma_el_p_h
from .sigma.sigmav_cx_h0 import sigmav_cx_h0

from .common import constants as CONST
from .common.JH_Coef import JH_Coef
from .common.Kinetic_H import *

@dataclass
class ComputeFlags():
    do_sigv: bool = True
    do_ni: bool = True
    do_fH2_moments: bool = True
    do_Alpha_CX: bool = True
    do_SIG_CX: bool = True
    do_Alpha_H_H2: bool = True
    do_SIG_H_H2: bool = True
    do_SIG_H_H: bool = True
    do_Alpha_H_P: bool = True
    do_SIG_H_P: bool = True
    do_v_v2: bool = True

class KineticH():

    #	Internal Debug switches
    CI_Test = 1
    Do_Alpha_CX_Test = 0

    #	Internal Tolerances
    DeltaVx_tol = .01
    Wpp_tol = .001

    # Theta-prime Coordinate
    ntheta = 5 # use 5 theta mesh points for theta integration
    dtheta = np.ones(ntheta)/ntheta
    cos_theta = np.cos(np.pi*(np.arange(ntheta)/ntheta + 0.5/ntheta))

    #   Internal Print Formatting
    prompt = 'Kinetic_H => '


    def __init__(self, mesh: KineticMesh, mu: int, vxi: NDArray, fHBC: NDArray, GammaxHBC: float, jh_coeffs: JH_Coef = None,
                 debrief = 0, compute_errors = 0):

        # Configuration Options
        self.COLLISIONS = get_config()['collisions']
        self.ion_rate_option = get_config()['kinetic_h']['ion_rate']

        # Run Settings
        self.debrief = debrief

        # Main attributes
        self.mesh = mesh
        self.mu = mu
        self.vxi = vxi
        self.fHBC = fHBC
        self.GammaxHBC = GammaxHBC

        # Shorthand sizes for main mesh variables
        self.nvr = mesh.vr.size
        self.nvx = self.mesh.vx.size
        self.nx = self.mesh.x.size

        self.vx_neg = np.nonzero(self.mesh.vx < 0)[0]
        self.vx_pos = np.nonzero(self.mesh.vx > 0)[0]
        self.vx_zero = np.nonzero(self.mesh.vx == 0)[0]

        # Variables for internal use
        self.vth = np.sqrt((2*CONST.Q*self.mesh.Tnorm) / (self.mu*CONST.H_MASS))
        self.fHBC_input = self._compute_fBHC_input()
        #	Vr^2-2*Vx^2
        self.vr2_2vx2_2D = np.asarray([(vr**2) - 2*(self.mesh.vx**2) for vr in self.mesh.vr])
        #   Differential Values
        differential = VSpace_Differentials(self.mesh.vr, self.mesh.vx)
        self.dvr_volume = differential.dvr_vol
        self.dvx = differential.dvx

        # Common Blocks
        self.Input = Kinetic_H_Input()
        self.Internal = Kinetic_H_Internal()
        self.Output = Kinetic_H_Output(self.nx)
        self.H2_Moments = Kinetic_H_H2_Moments()
        self.Errors = Kinetic_H_Errors()
        self.JH_Coefficients = jh_coeffs

        self._test_init_parameters()

        self.compute_flags = ComputeFlags()

        # Initial Computations
        # Some may not be used depending on inputs
        self._compute_grid()
        self._compute_protons()
        self._compute_sigv()
        self._compute_v_v2()
        self._compute_sig_cx()
        self._compute_sig_h_h()
        self._compute_sig_h_h2()
        self._compute_sig_h_p()

        if compute_errors:
            self._compute_vbar_error()


        return
    

    #def _determine_compute_flags():

    
    
    def run_generation(self, fH2: NDArray = None, fSH: NDArray = None, fH: NDArray = None, nHP: NDArray = None, THP: NDArray = None, 
              truncate: float = 1e-4, max_gen = 50, ni_correct = 0, compute_errors = 0, recomb = True, plot = 0, debug = 0, debrief = 0, pause = 0):
        '''
        Solves a 1-D spatial, 2-D velocity kinetic neutral transport 
        problem for atomic hydrogen or deuterium (H)

        Parameters
        ----------
            mesh : KineticMesh
                Mesh data for h kinetic procedure, must be of type 'h'
                Includes coordinate data and temperature/density profiles
            mu : int
                1=hydrogen, 2=deuterium
            vxi : ndarray
                flow speed profile (m/s)
            fHBC : ndarray
                2D array, input boundary condition. Specifies shape of atom velocity distribution (fH) at x=0
            GammaxHBC : float
                Desired neutral atom flux density at x=0 (m^-2 s^-1)
            fH2 : ndarray, default=None
                3D array, molecular distribution function. If None, H-H2 collisions are not computed
            fSH : ndarray, defualt=None
                Source velocity distribution function. If None, zero array is used
            fH : ndarray, default=None
                3D array, atomic distribution function. If None, zero array is used
            nHP : ndarray, defualt=None
                Molecular ion density profile (m^-3). If None, zero array is used
            THP : ndarray, defualt=None
                Molecular ion temperature profile (m^-3). If None, array of 3.0 used
            jh_coeffs : JH_Coef, defualt=None
                Common blocks used to pass data for JH methods
                NOTE Consider changing this, program will currently fail if not set
            KH : Kinetic_H_Common, default=None
                Common blocks used to pass data.
                NOTE Consider changing this, program will currently fail if not set
            truncate : float, default=1.0e-4
                Convergence threshold for generations
            max_gen : int, default=50
                Max number of generations
            ni_correct : bool, default=False
                If true, Corrects hydrogen ion density according to quasineutrality: ni=ne-nHp
            compute_errors : bool, default=False
                If true, compute error estimates
            recomb : bool, default=True
                If true, includes recombination as a source of atomic neutrals in the algorithm
            plot : int, default=0
                - 0=no plots
                - 1=summary plots
                - 2=detail plots
                - 3=very detailed plots
            debug : int, default=0
                - 0=do not execute debug code
                - 1=summary debug
                - 2=detail debug
                - 3=very detailed debug
            debrief : int, default=0
                - 0=do not print
                - 1=print summary information
                - 2=print detailed information
            pause : bool, default=False
                If true, pause between plots

        Returns
        -------
        fH,nH,GammaxH,VxH,pH,TH,qxH,qxH_total,NetHSource,Sion,QH,RxH,QH_total,AlbedoH,WallH

            fH : ndarray
                3D array, atomic distribution function.
            nH : ndarray, defualt=None
                Neutral atom density profile (m^-3).
            GammaxH : ndarray
                Neutral flux profile (m^-2 s^-1)
            VxH : ndarray
                Neutral velocity profile (m s^-1)
            pH : ndarray
                Neutral pressure (eV m^-2)
            TH : ndarray
                Neutral temperature profile (eV)
            qxH : ndarray
                Neutral random heat flux profile (watts m^-2)
            qxH_total : ndarray
                Total neutral heat flux profile (watts m^-2)
            NetHSource : ndarray
                Net H0 source (m^-3 s^-1)
            Sion : ndarray
                H ionization rate (m^-3 s^-1)
            QH : ndarray
                Rate of net thermal energy transfer into neutral atoms (watts m^-3)
            RxH : ndarray
                Rate of x momentum transfer to neutral atoms (N m^-2)
            QH_total : ndarray
                Net rate of total energy transfer into neutral atoms (watts m^-3)
            AlbedoH : float
                Ratio of atomic particle flux with Vx < 0 divided by particle flux with Vx > 0 at x=0
            WallH : ndarray
                Atomic sink rate from interation with 'side walls' (m^-3 s^-1)

        Notes
        -------
            This subroutine is part of the "KN1D" atomic and molecular neutral transport code.

            This subroutine solves a 1-D spatial, 2-D velocity kinetic neutral transport 
            problem for atomic hydrogen (H) or deuterium by computing successive generations of 
            charge exchange and elastic scattered neutrals. The routine handles electron-impact 
            ionization, proton-atom charge exchange, radiative recombination, and elastic
            collisions with hydrogenic ions, neutral atoms, and molecules.

            The positive vx half of the atomic neutral distribution function is inputted at x(0) 
            (with arbitrary normalization) and the desired flux of hydrogen atoms entering the slab,
            at x(0) is specified. Background profiles of plasma ions, (e.g., Ti(x), Te(x), n(x), vxi(x),...)
            molecular ions, (nHP(x), THP(x)), and molecular distribution function (fH) are inputted.

            Optionally, the hydrogen source velocity distribution function is also inputted.
            (The H source and fH2 distribution functions can be computed using procedure 
            "Kinetic_H2.pro".) The code returns the atomic hydrogen distribution function, fH(vr,vx,x) 
            for all vx, vr, and x of the specified vr,vx,x grid.

            Since the problem involves only the x spatial dimension, all distribution functions
            are assumed to have rotational symmetry about the vx axis. Consequently, the distributions
            only depend on x, vx and vr where vr =sqrt(vy^2+vz^2)

            History:

                B. LaBombard   First coding based on Kinetic_Neutrals.pro 		22-Dec-2000

                For more information, see write-up: "A 1-D Space, 2-D Velocity, Kinetic 
                Neutral Transport Algorithm for Hydrogen Atoms in an Ionizing Plasma", B. LaBombard

            Variable names contain characters to help designate species -
            atomic neutral (H), molecular neutral (H2), molecular ion (HP), proton (i) or (P)
        '''

        # Override settings for debug
        if debug > 0:
            #plot = np.max(plot, 1)
            self.debrief = np.maximum(self.debrief, 1)
            pause = 1


        # --- Initialize inputs ---

        if fH2 is None:
            fH2 = np.zeros((self.nvr, self.nvx, self.nx))
        if fSH is None:
            fSH = np.zeros((self.nvr, self.nvx, self.nx))
        if nHP is None:
            nHP = np.zeros(self.nx)
        if THP is None:
            THP = np.full(self.nx, 1.0)
        if fH is None:
            fH = np.zeros((self.nvr,self.nvx,self.nx))
        self._test_input_parameters(fH2, fSH, nHP, THP, fH)

        self.H2_Moments.nH2 = np.zeros(self.nx)
        self.H2_Moments.VxH2 = np.zeros(self.nx)
        self.H2_Moments.TH2 = np.full(self.nx, 1.0)

        # If fH2 is zero, then turn off elastic H2 <-> H collisions
        H_H2_EL = self.COLLISIONS['H2_H_EL']
        if np.sum(fH2) <= 0:
            H_H2_EL = 0

        # Scale input molecular distribution function to agree with desired flux
        gamma_input = 1.0
        if abs(self.GammaxHBC) > 0:
            gamma_input = self.vth*np.sum(self.dvr_volume*(self.fHBC_input @ (self.mesh.vx*self.dvx)))
        ratio = abs(self.GammaxHBC)/gamma_input
        fHBC_input = self.fHBC_input*ratio
        if abs(ratio - 1) > 0.01*truncate:
            self.fHBC = fHBC_input
        fH[:,self.vx_pos,0] = fHBC_input[:,self.vx_pos]


        # --- Compute Needed Variables---

        New_Molecular_Ions = 1
        if (self.Input.nHP_s is not None) and np.array_equal(self.Input.nHP_s, nHP) and np.array_equal(self.Input.THP_s, THP):
            New_Molecular_Ions = 0

        New_fH2 = 1
        if (self.Input.fH2_s is not None) and np.array_equal(self.Input.fH2_s, fH2):
            New_fH2 = 0

        New_H_Seed = 1
        if (self.Input.fH_s is not None) and  np.array_equal(self.Input.fH_s, fH):
            New_H_Seed = 0

        if debug > 0:
            print("Kinetic H Settings")
            print("H_H_EL", self.COLLISIONS['H_H_EL'])
            print("H_P_EL", self.COLLISIONS['H_P_EL'])
            print("H_H2_EL", self.COLLISIONS['H2_H_EL'])
            print("H_P_CX", self.COLLISIONS['H_P_CX'])
            print("Simple_CX", self.COLLISIONS['SIMPLE_CX'])
            print("New_Molecular_Ions", New_Molecular_Ions)
            print("New_fH2", New_fH2)
            print("New_H_Seed", New_H_Seed)
            input()

        if New_H_Seed:
            self.Internal.MH_H_sum = np.zeros((self.nvr,self.nvx,self.nx))
            self.Internal.Delta_nHs = 1
        if New_fH2 and (np.sum(fH2) > 0.0):
            self._compute_fh2_moments(fH2)
        if New_Molecular_Ions:
            self._compute_ni(nHP, ni_correct)
        self._compute_sn(fSH, recomb)

        # Set up arrays for charge exchange and elastic collision computations, if needed
        if ((self.Internal.Alpha_CX is None) | New_Molecular_Ions) and self.COLLISIONS['H_P_CX']:
            self._compute_alpha_cx()
        if ((self.Internal.Alpha_H_H2 is None) | New_fH2) and H_H2_EL:
            self._compute_alpha_h_h2(fH2)
        if ((self.Internal.Alpha_H_P is None) | New_Molecular_Ions) and self.COLLISIONS['H_P_EL']:
            self._compute_alpha_h_p()

        #	Compute nH
        nH = np.zeros(self.nx)
        for k in range(self.nx):
            nH[k] = np.sum(self.dvr_volume*(fH[:,:,k] @ self.dvx))

        #	Compute Side-Wall collision rate
        gamma_wall = np.zeros((self.nvr,self.nvx,self.nx))
        for k in range(self.nx):
            if self.mesh.PipeDia[k] > 0:
                for j in range(self.nvx):
                    gamma_wall[:,j,k] = 2*self.mesh.vr / self.mesh.PipeDia[k]


        # --- Iteration ---

        do_fH_Iterate = True

        #	This is the entry point for fH iteration.
        #	Save 'seed' values for comparison later

        #	Set iteration scheme
        fH_iterate = 0
        if (self.COLLISIONS['H_H_EL'] != 0) or (self.COLLISIONS['H_P_EL'] != 0) or (H_H2_EL != 0): 
            fH_iterate = 1

        fH_generations = 0
        if (fH_iterate != 0) or (self.COLLISIONS['H_P_CX'] != 0): 
            fH_generations = 1

        fHG = np.zeros((self.nvr,self.nvx,self.nx))
        NHG = np.zeros((self.nx,max_gen+1))
        Omega_H_P = np.zeros(self.nx)
        Omega_H_H2 = np.zeros(self.nx)
        Omega_H_H = np.zeros(self.nx)
        while do_fH_Iterate: #NOTE Alpha_CX done before here, but done inside iteration in kh2, does it change per iteration? Is this an error?
            do_fH_Iterate = False
            nHs = copy.copy(nH)

            #	Compute Omega values if nH is non-zero
            ii = np.argwhere(nH <= 0)
            if ii.size <= 0:

                #	Compute VxH
                VxH = np.zeros(self.nx)
                if self.COLLISIONS['H_P_EL'] or H_H2_EL or self.COLLISIONS['H_H_EL']:
                    for k in range(self.nx):
                        VxH[k] = self.vth*np.sum(self.dvr_volume*(fH[:,:,k] @ (self.mesh.vx*self.dvx))) / nH[k]

                #	Compute Omega_H_P for present fH and Alpha_H_P if H_P elastic collisions are included
                if self.COLLISIONS['H_P_EL']:
                    if self.debrief > 1:
                        print(self.prompt+'Computing Omega_H_P')
                    for k in range(self.nx):
                        DeltaVx = (VxH[k] - self.vxi[k]) / self.vth
                        MagDeltaVx = np.maximum(abs(DeltaVx), self.DeltaVx_tol)
                        DeltaVx = np.sign(DeltaVx)*MagDeltaVx
                        Omega_H_P[k] = np.sum(self.dvr_volume*((self.Internal.Alpha_H_P[:,:,k]*fH[:,:,k]) @ self.dvx)) / (nH[k]*DeltaVx)
                    Omega_H_P = np.maximum(Omega_H_P, 0)

                #	Compute Omega_H_H2 for present fH and Alpha_H_H2 if H_H2 elastic collisions are included

                if H_H2_EL:
                    if self.debrief > 1:
                        print(self.prompt+'Computing Omega_H_H2')
                    for k in range(self.nx):
                        DeltaVx = (VxH[k] - self.H2_Moments.VxH2[k]) / self.vth
                        MagDeltaVx = np.maximum(abs(DeltaVx), self.DeltaVx_tol)
                        DeltaVx = np.sign(DeltaVx)*MagDeltaVx
                        # print("Mag", fH[:,:,k].T)
                        # input()
                        Omega_H_H2[k] = np.sum(self.dvr_volume*((self.Internal.Alpha_H_H2[:,:,k]*fH[:,:,k]) @ self.dvx)) / (nH[k]*DeltaVx)
                    Omega_H_H2 = np.maximum(Omega_H_H2, 0)

                #	Compute Omega_H_H for present fH if H_H elastic collisions are included

                if self.COLLISIONS['H_H_EL']:
                    if self.debrief > 1:
                        print(self.prompt+'Computing Omega_H_H')
                    Wperp_paraH = np.zeros(self.nx)
                    vr2_2vx_ran2 = np.zeros((self.nvr,self.nvx))
                    if np.sum(self.Internal.MH_H_sum) <= 0:
                        for k in range(self.nx):
                            for i in range(self.nvr):
                                vr2_2vx_ran2[i,:] = self.mesh.vr[i]**2 - 2*((self.mesh.vx - (VxH[k]/self.vth))**2)
                            Wperp_paraH[k] = np.sum(self.dvr_volume*((vr2_2vx_ran2*fH[:,:,k]) @ self.dvx)) / nH[k]
                    else:
                        for k in range(self.nx):
                            M_fH = self.Internal.MH_H_sum[:,:,k] - fH[:,:,k]
                            Wperp_paraH[k] = -np.sum(self.dvr_volume*((self.vr2_2vx2_2D*M_fH) @ self.dvx)) / nH[k]
                    for k in range(self.nx):
                        Work = fH[:,:,k].reshape((self.nvr*self.nvx), order='F')
                        Alpha_H_H = (self.Internal.SIG_H_H @ Work).reshape((self.nvr,self.nvx), order='F')
                        Wpp = Wperp_paraH[k]
                        MagWpp = np.maximum(np.abs(Wpp), self.Wpp_tol)
                        Wpp = np.sign(Wpp)*MagWpp
                        Omega_H_H[k] = np.sum(self.dvr_volume*((Alpha_H_H*Work.reshape((self.nvr,self.nvx), order='F')) @ self.dvx)) / (nH[k]*Wpp)
                    Omega_H_H = np.maximum(Omega_H_H, 0)

            #	Total Elastic scattering frequency
            Omega_EL = Omega_H_P + Omega_H_H2 + Omega_H_H

            #	Total collision frequency

            alpha_c = np.zeros((self.nvr,self.nvx,self.nx))
            if self.COLLISIONS['H_P_CX']:
                for k in range(self.nx):
                    alpha_c[:,:,k] = self.Internal.Alpha_CX[:,:,k] + self.Internal.alpha_ion[k] + Omega_EL[k] + gamma_wall[:,:,k]
            else:
                for k in range(self.nx):
                    alpha_c[:,:,k] = self.Internal.alpha_ion[k] + Omega_EL[k] + gamma_wall[:,:,k]

            #	Test x grid spacing based on Eq.(27) in notes
            if self.debrief>1:
                print(self.prompt+'Testing x grid spacing')
            Max_dx = np.full(self.nx, 1e32)
            for k in range(self.nx):
                for j in self.vx_pos:
                    Max_dx[k] = np.minimum(Max_dx[k], min(2*self.mesh.vx[j] / alpha_c[:,j,k]))

            dx = np.roll(self.mesh.x,-1) - self.mesh.x
            Max_dxL = Max_dx[0:self.nx-1]
            Max_dxR = Max_dx[1:self.nx]
            Max_dx = np.minimum(Max_dxL, Max_dxR)
            ilarge = np.argwhere(Max_dx < dx[0:self.nx-1])

            if ilarge.size > 0:
                print(self.prompt+'x mesh spacing is too large!') #NOTE Check Formatting
                debug = 1
                out = ""
                jj = 0

                #	Not sure the output is formatted correctly

                print(' \t    x(k+1)-x(k)   Max_dx(k)\t   x(k+1)-x(k)   Max_dx(k)\t   x(k+1)-x(k)   Max_dx(k)\t   x(k+1)-x(k)   Max_dx(k)\t   x(k+1)-x(k)   Max_dx(k)')
                for ii in range(ilarge.size):
                    jj += 1
                    out += ((str(ilarge[ii])+' \t')[:8]+(str(self.mesh.x[ilarge[ii]+1]-self.mesh.x[ilarge[ii]])+'        ')[:6]+'        '+str(Max_dx[ilarge[ii]])[:4]+'\t')
                    if jj>4:
                        print(out)
                        jj = 0
                        out = "\t"
                if jj>0:
                    print(out)
                raise Exception("x mesh spacing is too large")

            #	Define parameters Ak, Bk, Ck, Dk, Fk, Gk

            Ak = np.zeros((self.nvr,self.nvx,self.nx))
            Bk = np.zeros((self.nvr,self.nvx,self.nx))
            Ck = np.zeros((self.nvr,self.nvx,self.nx))
            Dk = np.zeros((self.nvr,self.nvx,self.nx))
            Fk = np.zeros((self.nvr,self.nvx,self.nx))
            Gk = np.zeros((self.nvr,self.nvx,self.nx))

            for k in range(0, self.nx-1):
                for j in self.vx_pos: # double check some of the ranges in for statements I might have some typos
                    denom = 2*self.mesh.vx[j] + (self.mesh.x[k+1] - self.mesh.x[k])*alpha_c[:,j,k+1]
                    Ak[:,j,k] = (2*self.mesh.vx[j] - (self.mesh.x[k+1] - self.mesh.x[k])*alpha_c[:,j,k]) / denom
                    Bk[:,j,k] = (self.mesh.x[k+1] - self.mesh.x[k]) / denom
                    Fk[:,j,k] = (self.mesh.x[k+1] - self.mesh.x[k])*(self.Internal.Sn[:,j,k+1]+self.Internal.Sn[:,j,k]) / denom
            for k in range(1, self.nx):
                for j in self.vx_neg:
                    denom = -2*self.mesh.vx[j] + (self.mesh.x[k] - self.mesh.x[k-1])*alpha_c[:,j,k-1]
                    Ck[:,j,k] = (-2*self.mesh.vx[j] - (self.mesh.x[k] - self.mesh.x[k -1])*alpha_c[:,j,k]) / denom
                    Dk[:,j,k] = (self.mesh.x[k] - self.mesh.x[k-1]) / denom
                    Gk[:,j,k] = (self.mesh.x[k] - self.mesh.x[k-1])*(self.Internal.Sn[:,j,k]+self.Internal.Sn[:,j,k-1]) / denom
                            
            #	Compute first-flight (0th generation) neutral distribution function
            Beta_CX_sum = np.zeros((self.nvr,self.nvx,self.nx))
            MH_P_sum = np.zeros((self.nvr,self.nvx,self.nx))
            MH_H2_sum = np.zeros((self.nvr,self.nvx,self.nx))
            self.Internal.MH_H_sum = np.zeros((self.nvr,self.nvx,self.nx))
            igen = 0
            if self.debrief > 0:
                print(self.prompt+'Computing atomic neutral generation#'+sval(igen))
            fHG[:,self.vx_pos,0] = fH[:,self.vx_pos,0]
            for k in range(self.nx-1):
                fHG[:,self.vx_pos,k+1] = fHG[:,self.vx_pos,k]*Ak[:,self.vx_pos,k] + Fk[:,self.vx_pos,k]
            for k in range(self.nx-1,0,-1):
                fHG[:,self.vx_neg,k-1] = fHG[:,self.vx_neg,k]*Ck[:,self.vx_neg,k] + Gk[:,self.vx_neg,k]
                    
            #	Compute first-flight neutral density profile
            for k in range(self.nx):
                NHG[k,igen] = np.sum(self.dvr_volume*(fHG[:,:,k] @ self.dvx))

            # NOTE Add plotting once program is working

            #	Set total atomic neutral distribution function to first flight generation

            fH = copy.copy(fHG)
            nH = NHG[:,0]

    # next_generation #########################################################################################################################################################################
            while True:
                if igen+1 > max_gen or fH_generations == 0:
                    if self.debrief > 0:
                        print(self.prompt+'Completed '+sval(max_gen)+' generations. Returning present solution...')
                    break
                igen += 1
                if self.debrief > 0:
                    print(self.prompt+'Computing atomic neutral generation#'+sval(igen))

                #	Compute Beta_CX from previous generation

                Beta_CX = np.zeros((self.nvr,self.nvx,self.nx))
                if self.COLLISIONS['H_P_CX']:
                    if self.debrief>1:
                        print(self.prompt+'Computing Beta_CX')

                    if self.COLLISIONS['SIMPLE_CX']:
                        #	Option (B): Compute charge exchange source with assumption that CX source 
                        #		neutrals have ion distribution function
                        for k in range(self.nx):
                            Beta_CX[:,:,k] = self.Internal.fi_hat[:,:,k]*np.sum(self.dvr_volume*((self.Internal.Alpha_CX[:,:,k]*fHG[:,:,k]) @ self.dvx))
                    else:
                        #	Option (A): Compute charge exchange source using fH and vr x sigma x v_v at 
                        #		each velocity mesh point
                        for k in range(self.nx):
                            Work = fHG[:,:,k]
                            Beta_CX[:,:,k] = self.Internal.ni[k]*self.Internal.fi_hat[:,:,k]*(self.Internal.SIG_CX @ Work)

                    #	Sum charge exchange source over all generations
                    Beta_CX_sum += Beta_CX

                #	Compute MH from previous generation
                MH_H = np.zeros((self.nvr,self.nvx,self.nx))
                MH_P = np.zeros((self.nvr,self.nvx,self.nx))
                MH_H2 = np.zeros((self.nvr,self.nvx,self.nx))
                OmegaM = np.zeros((self.nvr,self.nvx,self.nx))
                VxHG = np.zeros(self.nx)
                THG = np.zeros(self.nx)
                if self.COLLISIONS['H_H_EL'] or self.COLLISIONS['H_P_EL'] or H_H2_EL:

                    #	Compute VxHG, THG
                    vr2vx2_ran2 = np.zeros((self.nvr,self.nvx))
                    for k in range(0, self.nx):
                        VxHG[k] = self.vth*np.sum(self.dvr_volume*(fHG[:,:,k] @ (self.mesh.vx*self.dvx))) / NHG[k,igen-1]
                        for i in range(0, self.nvr):
                            vr2vx2_ran2[i,:] = self.mesh.vr[i]**2 + (self.mesh.vx - VxHG[k]/self.vth)**2
                        THG[k] = (self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*((vr2vx2_ran2*fHG[:,:,k]) @ self.dvx)) / (3*CONST.Q*NHG[k,igen-1])

                    if self.COLLISIONS['H_H_EL']:
                        if self.debrief > 1:
                            print(self.prompt+'Computing MH_H')

                        #	Compute MH_H 
                        vx_shift = VxHG
                        Tmaxwell = THG
                        mol = 1
                        Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
                        for k in range(self.nx):
                            MH_H[:,:,k] = Maxwell[:,:,k]*NHG[k,igen-1]
                            OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H_H[k]*MH_H[:,:,k]
                        self.Internal.MH_H_sum += MH_H

                    if self.COLLISIONS['H_P_EL']:
                        if self.debrief>1:
                            print(self.prompt+'Computing MH_P')

                        #	Compute MH_P 
                        vx_shift = (VxHG+self.vxi)/2
                        Tmaxwell = THG + (2/4)*(self.mesh.Ti - THG + self.mu*CONST.H_MASS*((self.vxi - VxHG)**2) / (6*CONST.Q))
                        mol = 1
                        Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
                        for k in range(self.nx):
                            MH_P[:,:,k] = Maxwell[:,:,k]*NHG[k,igen-1]
                            OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H_P[k]*MH_P[:,:,k]
                        MH_P_sum += MH_P

                    if H_H2_EL:
                        if self.debrief>1:
                            print(self.prompt+'Computing MH_H2')

                        #	Compute MH_H2
                        vx_shift = (VxHG + 2*self.H2_Moments.VxH2)/3
                        Tmaxwell = THG + (4./9.)*(self.H2_Moments.TH2 - THG + 2*self.mu*CONST.H_MASS*((self.H2_Moments.VxH2 - VxHG)**2) / (6*CONST.Q))
                        mol = 1
                        Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
                        
                        for k in range(self.nx):
                            MH_H2[:,:,k] = Maxwell[:,:,k]*NHG[k,igen-1]
                            OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H_H2[k]*MH_H2[:,:,k]
                        MH_H2_sum += MH_H2

                #	Compute next generation atomic distribution

                fHG[:] = 0
                for k in range(0, self.nx-1):
                    fHG[:,self.vx_pos,k+1] = Ak[:,self.vx_pos,k]*fHG[:,self.vx_pos,k] + Bk[:,self.vx_pos,k]*(Beta_CX[:,self.vx_pos,k+1] + OmegaM[:,self.vx_pos,k+1] + Beta_CX[:,self.vx_pos,k] + OmegaM[:,self.vx_pos,k])
                for k in range(self.nx-1, 0, -1):
                    fHG[:,self.vx_neg,k-1] = Ck[:,self.vx_neg,k]*fHG[:,self.vx_neg,k] + Dk[:,self.vx_neg,k]*(Beta_CX[:,self.vx_neg,k-1] + OmegaM[:,self.vx_neg,k-1] + Beta_CX[:,self.vx_neg,k] + OmegaM[:,self.vx_neg,k])
                for k in range(0, self.nx):
                    NHG[k,igen] = np.sum(self.dvr_volume*(fHG[:,:,k] @ self.dvx))

                # NOTE Add plotting once program is working

                #	Add result to total neutral distribution function
                fH += fHG
                nH += NHG[:,igen]

                #	Compute 'generation error': Delta_nHG=max(NHG(*,igen)/max(nH))
                #		and decide if another generation should be computed
                Delta_nHG = max(NHG[:,igen]/max(nH))
                if (Delta_nHG < truncate) or (fH_iterate and (Delta_nHG < 0.003*self.Internal.Delta_nHs)):
                    #	If fH 'seed' is being iterated, then do another generation until the 'generation error'
                    #		is less than 0.003 times the 'seed error' or is less than TRUNCATE
                    break

    # fH2_done #########################################################################################################################################################################
            
            # NOTE Add plotting once program is working

            #	Compute H density profile
            for k in range(0, self.nx):
                nH[k] = np.sum(self.dvr_volume*(fH[:,:,k] @ self.dvx))

            if fH_iterate:

                #	Compute 'seed error': Delta_nHs=(|nHs-nH|)/max(nH) 
                #		If Delta_nHs is greater than 10*truncate then iterate fH
                self.Internal.Delta_nHs = np.max(np.abs(nHs - nH))/np.max(nH)
                if self.Internal.Delta_nHs > 10*truncate:
                    do_fH_Iterate = True 

        #	Update Beta_CX_sum using last generation
        if self.COLLISIONS['H_P_CX']:
            if self.debrief > 1:
                print(self.prompt, 'Computing Beta_CX')
            if self.COLLISIONS['SIMPLE_CX']:
                # Option (B): Compute charge exchange source with assumption that CX source neutrals have
                # ion distribution function
                for k in range(0, self.nx):
                    Beta_CX[:,:,k] = self.Internal.fi_hat[:,:,k]*np.sum(self.dvr_volume*(self.Internal.Alpha_CX[:,:,k]*fHG[:,:,k] @ self.dvx))
            else:
                # Option (A): Compute charge exchange source using fH2 and vr x sigma x v_v at each velocity mesh point
                for k in range(0, self.nx):
                    Work = fHG[:,:,k]
                    Beta_CX[:,:,k] = self.Internal.ni[k]*self.Internal.fi_hat[:,:,k]*(self.Internal.SIG_CX @ Work)
            Beta_CX_sum = Beta_CX_sum + Beta_CX
                
        #	Update MH_*_sum using last generation
        MH_H2 = np.zeros((self.nvr,self.nvx,self.nx))
        MH_P = np.zeros((self.nvr,self.nvx,self.nx))
        MH_H = np.zeros((self.nvr,self.nvx,self.nx))
        OmegaM = np.zeros((self.nvr,self.nvx,self.nx))
        VxHG = np.zeros(self.nx)
        THG = np.zeros(self.nx)
        if self.COLLISIONS['H_H_EL'] or self.COLLISIONS['H_P_EL'] or H_H2_EL: 
            # Compute VxH2G, TH2G
            vr2vx2_ran2 = np.zeros((self.nvr,self.nvx))
            for k in range(0, self.nx):
                VxHG[k] = self.vth*np.sum(self.dvr_volume*(fHG[:,:,k] @ (self.mesh.vx*self.dvx))) / NHG[k,igen]
                for i in range(0, self.nvr):
                    vr2vx2_ran2[i,:] = self.mesh.vr[i]**2 + (self.mesh.vx - VxHG[k]/self.vth)**2
                THG[k] = (self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*((vr2vx2_ran2*fHG[:,:,k]) @ self.dvx)) / (3*CONST.Q*NHG[k,igen])

            if self.COLLISIONS['H_H_EL']:
                if self.debrief > 1: 
                    print(self.prompt, 'Computing MH_H')
                # Compute MH_H
                vx_shift = VxHG
                Tmaxwell = np.copy(THG)
                mol = 1
                Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
                
                for k in range(0, self.nx):
                    MH_H[:,:,k] = Maxwell[:,:,k]*NHG[k,igen]
                    OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H_H[k]*MH_H[:,:,k]
                self.Internal.MH_H_sum = self.Internal.MH_H_sum + MH_H

            if self.COLLISIONS['H_P_EL']:
                if self.debrief > 1:
                    print(self.prompt, 'Computing MH_P')
                # Compute MH_P
                vx_shift = (VxHG + self.vxi)/2
                Tmaxwell = THG + (2/4)*(self.mesh.Ti - THG + self.mu*CONST.H_MASS*((self.vxi - VxHG)**2) / (6*CONST.Q))
                mol = 1
                Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
                
                for k in range(0, self.nx):
                    MH_P[:,:,k] = Maxwell[:,:,k]*NHG[k,igen]
                    OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H_P[k]*MH_P[:,:,k]
                MH_P_sum = MH_P_sum + MH_P

                if H_H2_EL: #NOTE Not Tested Yet
                    if self.debrief > 1:
                        print(self.prompt, 'Computing MH_H2')
                    # Compute MH_H
                    vx_shift = (VxHG + 2*self.H2_Moments.VxH2)/3
                    Tmaxwell = THG + (4/9)*(self.H2_Moments.TH2 - THG + 2*self.mu*CONST.H_MASS*((self.H2_Moments.VxH2 - VxHG)**2) / (6*CONST.Q))
                    mol = 1
                    Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
                    for k in range(0, self.nx):
                        MH_H2[:,:,k] = Maxwell[:,:,k]*NHG[k,igen]
                        OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H_H2[k]*MH_H2[:,:,k]
                    MH_H2_sum = MH_H2_sum + MH_H2

        #	Compute remaining moments

        #NOTE In kinetic_h2, these are calculated in the iteration, does that matter?
        #	GammaxH - particle flux in x direction
        GammaxH = np.zeros(self.nx)
        for k in range(0, self.nx):
                GammaxH[k] = self.vth*np.sum(self.dvr_volume*(fH[:,:,k] @ (self.mesh.vx*self.dvx)))

        #	VxH - x velocity
        VxH = GammaxH / nH
        _VxH = VxH / self.vth

        #	magnitude of random velocity at each mesh point
        vr2vx2_ran = np.zeros((self.nvr,self.nvx,self.nx))
        for i in range(0, self.nvr):
            for k in range(0, self.nx):
                vr2vx2_ran[i,:,k] = self.mesh.vr[i]**2 + (self.mesh.vx - _VxH[k])**2

        #	pH - pressure 
        pH = np.zeros(self.nx)
        for k in range(self.nx):
            pH[k] = ((self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*((vr2vx2_ran[:,:,k]*fH[:,:,k]) @ self.dvx))) / (3*CONST.Q)

        #	TH - temperature
        TH = pH/nH

        #	piH_xx
        for k in range(self.nx):
            self.Output.piH_xx[k] = (((self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*(fH[:,:,k] @ (self.dvx*(self.mesh.vx - _VxH[k])**2)))) / CONST.Q) - pH[k]
        #	piH_yy
        for k in range(self.nx):
            self.Output.piH_yy[k] = (((self.mu*CONST.H_MASS)*self.vth**2*0.5*np.sum((self.dvr_volume*(self.mesh.vr**2))*(fH[:,:,k] @ self.dvx))) / CONST.Q) - pH[k]
        #	piH_zz
        self.Output.piH_zz = copy.copy(self.Output.piH_yy)
        #	qxH
        qxH = np.zeros(self.nx)
        for k in range(self.nx):
            qxH[k] = 0.5*(self.mu*CONST.H_MASS)*self.vth**3*np.sum(self.dvr_volume*((vr2vx2_ran[:,:,k]*fH[:,:,k]) @ (self.dvx*(self.mesh.vx - _VxH[k]))))

        #	C = RHS of Boltzman equation for total fH

        QH = np.zeros(self.nx)
        RxH = np.zeros(self.nx)
        NetHSource = np.zeros(self.nx)
        Sion = np.zeros(self.nx)
        WallH = np.zeros(self.nx)
        for k in range(self.nx):
            C = self.vth*(self.Internal.Sn[:,:,k] + Beta_CX_sum[:,:,k] - alpha_c[:,:,k]*fH[:,:,k] + \
                    Omega_H_P[k]*MH_P_sum[:,:,k] + Omega_H_H2[k]*MH_H2_sum[:,:,k] + Omega_H_H[k]*self.Internal.MH_H_sum[:,:,k])
            # print("C", C.T)
            # input()
            QH[k] = 0.5*(self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*((vr2vx2_ran[:,:,k]*C) @ self.dvx))
            RxH[k] = (self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_volume*(C @ (self.dvx*(self.mesh.vx - _VxH[k]))))
            NetHSource[k] = np.sum(self.dvr_volume*(C @ self.dvx))
            Sion[k] = self.vth*nH[k]*self.Internal.alpha_ion[k]
            self.Output.SourceH[k] = np.sum(self.dvr_volume*(fSH[:,:,k] @ self.dvx))
            WallH[k] = np.sum(self.dvr_volume*((gamma_wall[:,:,k]*fH[:,:,k]) @ self.dvx))

            if recomb:
                self.Output.SRecomb[k] = self.vth*self.Internal.ni[k]*self.Internal.Rec[k]
            else:
                self.Output.SRecomb[k] = 0

            if self.COLLISIONS['H_P_CX']:
                CCX = self.vth*(Beta_CX_sum[:,:,k] - self.Internal.Alpha_CX[:,:,k]*fH[:,:,k])
                self.Output.RxHCX[k] = (self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_volume*(CCX @ (self.dvx*(self.mesh.vx - _VxH[k]))))
                self.Output.EHCX[k] = 0.5*(self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*((self.Internal.vr2vx2[:,:,k]*CCX) @ self.dvx))

            if H_H2_EL:
                CH_H2 = self.vth*Omega_H_H2[k]*(MH_H2_sum[:,:,k] - fH[:,:,k])
                self.Output.RxH2_H[k] = (self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_volume*(CH_H2 @ (self.dvx*(self.mesh.vx - _VxH[k]))))
                self.Output.EH2_H[k] = 0.5*(self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*((self.Internal.vr2vx2[:,:,k]*CH_H2) @ self.dvx))

            if self.COLLISIONS['H_P_EL']:
                CH_P = self.vth*Omega_H_P[k]*(MH_P_sum[:,:,k] - fH[:,:,k])
                self.Output.RxP_H[k] = (self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_volume*(CH_P @ (self.dvx*(self.mesh.vx - _VxH[k]))))
                self.Output.EP_H[k] = 0.5*(self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*((self.Internal.vr2vx2[:,:,k]*CH_P) @ self.dvx))

            CW_H = -self.vth*(gamma_wall[:,:,k]*fH[:,:,k])
            self.Output.RxW_H[k] = (self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_volume*(CW_H @ (self.dvx*(self.mesh.vx - _VxH[k]))))
            self.Output.EW_H[k] = 0.5*(self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*((self.Internal.vr2vx2[:,:,k]*CW_H) @ self.dvx))
            
            if self.COLLISIONS['H_H_EL']:
                vr2_2vx_ran2 = np.zeros((self.nvr,self.nvx))
                CH_H = self.vth*Omega_H_H[k]*(self.Internal.MH_H_sum[:,:,k] - fH[:,:,k])
                for i in range(0, self.nvr):
                    vr2_2vx_ran2[i,:] = self.mesh.vr[i]**2 - 2*((self.mesh.vx - _VxH[k])**2)
                self.Output.Epara_PerpH_H[k] = -0.5*(self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*((vr2_2vx_ran2*CH_H) @ self.dvx))

        #	qxH_total
        qxH_total = (0.5*nH*(self.mu*CONST.H_MASS)*VxH*VxH + 2.5*pH*CONST.Q)*VxH + CONST.Q*self.Output.piH_xx*VxH + qxH

        #	QH_total
        QH_total = QH + RxH*VxH + 0.5*(self.mu*CONST.H_MASS)*NetHSource*VxH*VxH

        #	Albedo
        gammax_plus = self.vth*np.sum(self.dvr_volume*(fH[:,self.vx_pos,0] @ (self.mesh.vx[self.vx_pos]*self.dvx[self.vx_pos]))) 
        gammax_minus = self.vth*np.sum(self.dvr_volume*(fH[:,self.vx_neg,0] @ (self.mesh.vx[self.vx_neg]*self.dvx[self.vx_neg])))
        AlbedoH = 0.0e0
        if np.abs(gammax_plus) > 0:
            AlbedoH = -gammax_minus/gammax_plus

        if compute_errors:
            self._compute_final_errors(Sion, WallH, NetHSource, Beta_CX_sum, fH, H_H2_EL, MH_H2_sum, MH_P_sum, alpha_c, Omega_H_P, Omega_H_H2, Omega_H_H, qxH_total, QH_total, debug)

        # NOTE Add plotting once program is working

        self.Input.vx_s = self.mesh.vx
        self.Input.vr_s = self.mesh.vr
        self.Input.x_s = self.mesh.x
        self.Input.Tnorm_s = self.mesh.Tnorm
        self.Input.mu_s = self.mu
        self.Input.Ti_s = self.mesh.Ti
        self.Input.Te_s = self.mesh.Te
        self.Input.n_s = self.mesh.ne
        self.Input.vxi_s = self.vxi
        self.Input.fHBC_s = self.fHBC
        self.Input.GammaxHBC_s = self.GammaxHBC
        self.Input.PipeDia_s = self.mesh.PipeDia
        self.Input.fH2_s = fH2
        self.Input.fSH_s = fSH
        self.Input.nHP_s = nHP
        self.Input.THP_s = THP
        self.Input.fH_s = fH
        self.Input.Simple_CX_s = self.COLLISIONS['SIMPLE_CX']
        self.Input.Recomb_s = recomb
        self.Input.H_H_EL_s = self.COLLISIONS['H_H_EL']
        self.Input.H_P_EL_s = self.COLLISIONS['H_P_EL']
        self.Input.H_H2_EL_s = H_H2_EL
        self.Input.H_P_CX_s = self.COLLISIONS['H_P_CX']

        return fH,nH,GammaxH,VxH,pH,TH,qxH,qxH_total,NetHSource,Sion,QH,RxH,QH_total,AlbedoH,WallH
    

    def _compute_fBHC_input(self):
        '''
        Computes fHBC_input, helper function for initialization
        '''
        fHBC_input = np.zeros(self.fHBC.shape)
        fHBC_input[:,self.vx_pos] = self.fHBC[:,self.vx_pos]
        if np.sum(fHBC_input) <= 0.0 and abs(self.GammaxHBC) > 0:
            raise Exception(self.prompt+'Values for fHBC[:,:] with vx > 0 are all zero!')
        
        return fHBC_input
    

    def _test_init_parameters(self):
        '''
        Performs compatibility tests for passed parameters when initializing class
        '''

        dx = self.mesh.x - np.roll(self.mesh.x, 1)
        dx = dx[1:]
        notpos = np.argwhere(dx <= 0)
        if notpos.size > 0:
            raise Exception(self.prompt + 'x[*] must be increasing with index!')
        if self.nvx % 2 != 0:
            raise Exception(self.prompt + 'Number of elements in vx must be even!') 
        if self.mesh.Ti.size != self.nx:
            raise Exception(self.prompt + 'Number of elements in Ti and x do not agree!')
        if self.vxi is None:
            self.vxi = np.zeros(self.nx)
        if self.vxi.size != self.nx:
            raise Exception(self.prompt + 'Number of elements in vxi and x do not agree!')
        if self.mesh.Te.size != self.nx:
            raise Exception(self.prompt + 'Number of elements in Te and x do not agree!')
        if self.mesh.ne.size != self.nx:
            raise Exception(self.prompt + 'Number of elements in n and x do not agree!')
        if self.GammaxHBC is None:
            raise Exception(self.prompt + 'GammaxHBC is not defined!')
        if self.mesh.PipeDia is None:
            self.mesh.PipeDia = np.zeros(self.nx)
        if self.mesh.PipeDia.size != self.nx:
            raise Exception(self.prompt + 'Number of elements in PipeDia and x do not agree!')
        if len(self.fHBC[:,0]) != self.nvr:
            raise Exception(self.prompt + 'Number of elements in fHBC[:,0] and vr do not agree!')
        if len(self.fHBC[0,:]) != self.nvx:
            raise Exception(self.prompt + 'Number of elements in fHBC[0,:] and vx do not agree!')
        if np.sum(abs(self.mesh.vr)) <= 0:
            raise Exception(self.prompt+'vr is all 0!')
        ii = np.argwhere(self.mesh.vr <= 0)
        if ii.size > 0:
            raise Exception(self.prompt+'vr contains zero or negative element(s)!')
        if np.sum(abs(self.mesh.vx)) <= 0:
            raise Exception(self.prompt+'vx is all 0!')
        if np.sum(self.mesh.x) <= 0:
            raise Exception(self.prompt+'Total(x) is less than or equal to 0!')
        if self.mu is None:
            raise Exception(self.prompt+'mu is not defined!')
        if self.mu not in [1,2]:
            raise Exception(self.prompt+'mu must be 1 or 2!')
        
        if np.size(self.vx_neg) < 1:
            print(self.prompt+'vx contains no negative elements!')
        if np.size(self.vx_pos) < 1:
            print(self.prompt+'vx contains no positive elements!')
        if np.size(self.vx_zero) > 0:
            print(self.prompt+'vx contains one or more zero elements!')
        diff = np.nonzero(self.mesh.vx[self.vx_pos] != -np.flipud(self.mesh.vx[self.vx_neg]))[0]
        if diff.size > 0:
            raise Exception(self.prompt + " vx array elements are not symmetric about zero!")
        
        return
        
    
    def _test_input_parameters(self, fH2, fSH, nHP, THP, fH):
        '''
        Performs compatibility tests for passed parameters when calling run_generation
        '''

        if fH2[:,0,0].size != self.nvr:
            raise Exception(self.prompt+'Number of elements in fH2[:,0,0] and vr do not agree!')
        if fH2[0,:,0].size != self.nvx:
            raise Exception(self.prompt+'Number of elements in fH2[0,:,0] and vx do not agree!')
        if fH2[0,0,:].size != self.nx:
            raise Exception(self.prompt+'Number of elements in fH2[0,0,:] and x do not agree!')
        if fSH[:,0,0].size != self.nvr:
            raise Exception(self.prompt+'Number of elements in fSH[:,0,0] and vr do not agree!')
        if fSH[0,:,0].size != self.nvx:
            raise Exception(self.prompt+'Number of elements in fSH[0,:,0] and vx do not agree!')
        if fSH[0,0,:].size != self.nx:
            raise Exception(self.prompt+'Number of elements in fSH[0,0,:] and x do not agree!')
        if nHP.size != self.nx:
            raise Exception(self.prompt+'Number of elements in nHP and x do not agree!')
        if THP.size != self.nx:
            raise Exception(self.prompt+'Number of elements in nHP and x do not agree!')
        if fH[:,0,0].size != self.nvr:
            raise Exception(self.prompt+'Number of elements in fH[:,0,0] and vr do not agree!')
        if fH[0,:,0].size != self.nvx:
            raise Exception(self.prompt+'Number of elements in fH[0,:,0] and vx do not agree!')
        if fH[0,0,:].size != self.nx:
            raise Exception(self.prompt+'Number of elements in fH[0,0,:] and x do not agree!')
        
        return


    # ------ Compute Functions ------


    # --- init ---

    def _compute_grid(self):
        if self.debrief > 1:
            print(self.prompt+'Computing vr2vx2, vr2vx_vxi2, ErelH_P')

        # Magnitude of total normalized v^2 at each mesh point
        self.Internal.vr2vx2 = np.zeros((self.nvr,self.nvx,self.nx))
        for i in range(self.nvr):
            for k in range(self.nx):
                self.Internal.vr2vx2[i,:,k] = self.mesh.vr[i]**2 + self.mesh.vx**2

        # Magnitude of total normalized (v-vxi)^2 at each mesh point
        self.Internal.vr2vx_vxi2 = np.zeros((self.nvr,self.nvx,self.nx))
        for i in range(self.nvr):
            for k in range(self.nx):
                self.Internal.vr2vx_vxi2[i,:,k] = self.mesh.vr[i]**2 + (self.mesh.vx - self.vxi[k]/self.vth)**2

        # Atomic hydrogen ion energy in local rest frame of plasma at each mesh point
        self.Internal.ErelH_P = (0.5*CONST.H_MASS*self.Internal.vr2vx_vxi2*self.vth**2) / CONST.Q
        self.Internal.ErelH_P = np.maximum(self.Internal.ErelH_P, 0.1) # sigmav_cx does not handle neutral energies below 0.1 eV
        self.Internal.ErelH_P = np.minimum(self.Internal.ErelH_P, 2e4) # sigmav_cx does not handle neutral energies above 20 keV


    def _compute_protons(self):
        if self.debrief>1:
            print(self.prompt+'Computing Ti/mu at each mesh point')

        # Ti/mu at each mesh point
        self.Internal.Ti_mu = np.zeros((self.nvr,self.nvx,self.nx))
        for k in range(self.nx):
            self.Internal.Ti_mu[:,:,k] = self.mesh.Ti[k] / self.mu

        # Compute Fi_hat
        if self.debrief>1:
            print(self.prompt+'Computing fi_Hat')
        vx_shift = self.vxi
        Tmaxwell = self.mesh.Ti
        mol = 1
        self.Internal.fi_hat = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)


    def _compute_sigv(self):
        if self.debrief>1:
            print(self.prompt+'Computing sigv')

        #	Compute sigmav rates for each reaction with option to use rates
        #	from CR model of Johnson-Hinnov

        self.Internal.sigv = np.zeros((self.nx,3))

        #	Reaction R1:  e + H -> e + H(+) + e   (ionization)
        #NOTE Replace Use_Collrad_Ionization with constant for consistency across program, check with someone who knows what they are doing if this is correct
        if self.ion_rate_option == "collrad":
            self.Internal.sigv[:,1] = collrad_sigmav_ion_h0(self.mesh.ne, self.mesh.Te) # from COLLRAD code (DEGAS-2)
        #NOTE Replace JH with constant for consistency across program, check with someone who knows what they are doing if this is correct
        elif self.ion_rate_option == "jh":
            print("using in mesh")
            self.Internal.sigv[:,1] = jhs_coef(self.mesh.ne, self.mesh.Te, self.JH_Coefficients, no_null=True) # Johnson-Hinnov, limited Te range; fixed JHS_coef capitalization #NOTE Not tested yet
        else:
            self.Internal.sigv[:,1] = sigmav_ion_h0(self.mesh.Te) # from Janev et al., up to 20keV #NOTE Not Tested Yet
                
        #	Reaction R2:  e + H(+) -> H(1s) + hv  (radiative recombination)
        #NOTE Replace JH with constant for consistency across program, check with someone who knows what they are doing if this is correct
        if self.ion_rate_option == "jh":
            self.Internal.sigv[:,2] = jhalpha_coef(self.mesh.ne, self.mesh.Te, self.JH_Coefficients, no_null=True)
        else:
            self.Internal.sigv[:,2] = sigmav_rec_h1s(self.mesh.Te)

        #	H ionization rate (normalized by vth) = reaction 1
        self.Internal.alpha_ion = (self.mesh.ne*self.Internal.sigv[:,1]) / self.vth

        #	Recombination rate (normalized by vth) = reaction 2
        self.Internal.Rec = (self.mesh.ne*self.Internal.sigv[:,2]) / self.vth


    def _compute_v_v2(self):
        if self.debrief > 1:
            print(self.prompt+'Computing v_v2, v_v, vr2_vx2, and vx_vx')

        #	v_v2=(v-v_prime)^2 at each double velocity space mesh point, including theta angle
        self.Internal.v_v2 = np.zeros((self.nvr,self.nvx,self.nvr,self.nvx,self.ntheta))

        #	vr2_vx2=0.125* [ vr2 + vr2_prime - 2*vr*vr_prime*cos(theta) - 2*(vx-vx_prime)^2 ]
        #		at each double velocity space mesh point, including theta angle
        self.Internal.vr2_vx2 = np.zeros((self.nvr,self.nvx,self.nvr,self.nvx,self.ntheta))
        for m in range(self.ntheta):
            for l in range(self.nvx):
                for k in range(self.nvr):
                    for i in range(self.nvr):
                        v_starter = self.mesh.vr[i]**2 + self.mesh.vr[k]**2 - 2*self.mesh.vr[i]*self.mesh.vr[k]*self.cos_theta[m]
                        self.Internal.v_v2[i,:,k,l,m] = v_starter + (self.mesh.vx[:] - self.mesh.vx[l])**2
                        self.Internal.vr2_vx2[i,:,k,l,m] = v_starter - 2*(self.mesh.vx[:] - self.mesh.vx[l])**2

        #	v_v=|v-v_prime| at each double velocity space mesh point, including theta angle
        self.Internal.v_v = np.sqrt(self.Internal.v_v2)

        #	vx_vx=(vx-vx_prime) at each double velocity space mesh point
        self.Internal.vx_vx = np.zeros((self.nvr,self.nvx,self.nvr,self.nvx))
        for j in range(self.nvx):
            for l in range(self.nvx):
                self.Internal.vx_vx[:,j,:,l] = self.mesh.vx[j] - self.mesh.vx[l]

        #	Set Vr'2pidVr'*dVx' for each double velocity space mesh point

        self.Internal.Vr2pidVrdVx = np.zeros((self.nvr,self.nvx,self.nvr,self.nvx))
        for k in range(self.nvr):
            self.Internal.Vr2pidVrdVx[:,:,k,:] = self.dvr_volume[k]
        for l in range(self.nvx):
            self.Internal.Vr2pidVrdVx[:,:,:,l] = self.Internal.Vr2pidVrdVx[:,:,:,l]*self.dvx[l]


    def _compute_sig_cx(self):
        if self.debrief>1:
            print(self.prompt+'Computing SIG_CX')

        #	Option (A) was selected: Compute SigmaV_CX from sigma directly.
        #	In preparation, compute SIG_CX for present velocity space grid, if it has not 
        #	already been computed with the present input parameters

        #	Compute sigma_cx * v_v at all possible relative velocities
        _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx, self.ntheta))
        _Sig[:] = (self.Internal.v_v*sigma_cx_h0(self.Internal.v_v2*(0.5*CONST.H_MASS*self.vth**2/CONST.Q))).reshape(_Sig.shape, order='F')

        #	Set SIG_CX = vr' x Integral{v_v*sigma_cx} 
        #		over theta=0,2pi times differential velocity space element Vr'2pidVr'*dVx'
        self.Internal.SIG_CX = np.zeros((self.nvr*self.nvx, self.nvr*self.nvx))
        self.Internal.SIG_CX[:] = (self.Internal.Vr2pidVrdVx*((_Sig @ self.dtheta).reshape(self.Internal.Vr2pidVrdVx.shape, order='F'))).reshape(self.Internal.SIG_CX.shape, order='F')

        #	SIG_CX is now vr' * sigma_cx(v_v) * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])


    def _compute_sig_h_h(self):
        if self.debrief>1:
            print(self.prompt+'Computing SIG_H_H')

        #	Compute SIG_H_H for present velocity space grid, if it is needed and has not already been computed with the present input parameters

        #	Compute sigma_H_H * vr2_vx2 * v_v at all possible relative velocities
        _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx,self.ntheta))
        _Sig[:] = (self.Internal.vr2_vx2*self.Internal.v_v*sigma_el_h_h(self.Internal.v_v2*(0.5*CONST.H_MASS*self.mu*self.vth**2/CONST.Q), vis=True) / 8).reshape(_Sig.shape, order='F')

        #	Note: For viscosity, the cross section for D -> D is the same function of center of mass energy as H -> H.

        #	Set SIG_H_H = vr' x Integral{vr2_vx2*v_v*sigma_H_H} over theta=0,2pi times differential velocity space element Vr'2pidVr'*dVx'
        self.Internal.SIG_H_H = np.zeros((self.nvr*self.nvx,self.nvr*self.nvx))
        self.Internal.SIG_H_H[:] = (self.Internal.Vr2pidVrdVx*(_Sig @ self.dtheta).reshape(self.Internal.Vr2pidVrdVx.shape, order='F')).reshape(self.Internal.SIG_H_H.shape, order='F')
        #	SIG_H_H is now vr' * sigma_H_H(v_v) * vr2_vx2 * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])


    def _compute_sig_h_h2(self):
        if self.debrief > 1:
            print(self.prompt+'Computing SIG_H_H2')

        #	Compute SIG_H_H2 for present velocity space grid, if it is needed and has not
        #		already been computed with the present input parameters

        #	Compute sigma_H_H2 * v_v at all possible relative velocities

        _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx,self.ntheta))
        _Sig[:] = (self.Internal.v_v*sigma_el_h_hh(self.Internal.v_v2*(0.5*CONST.H_MASS*self.vth**2/CONST.Q))).reshape(_Sig.shape, order='F')

        #	NOTE: using H energy here for cross-sections tabulated as H->H2

        #	Set SIG_H_H2 = vr' x vx_vx x Integral{v_v*sigma_H_H2} over theta=0,
        #		2pi times differential velocity space element Vr'2pidVr'*dVx'

        self.Internal.SIG_H_H2 = np.zeros((self.nvr*self.nvx,self.nvr*self.nvx))
        self.Internal.SIG_H_H2[:] = (self.Internal.Vr2pidVrdVx*self.Internal.vx_vx*(_Sig @ self.dtheta).reshape(self.Internal.Vr2pidVrdVx.shape, order='F')).reshape(self.Internal.SIG_H_H2.shape, order='F')

        #	SIG_H_H2 is now vr' *vx_vx * sigma_H_H2(v_v) * v_v 
        #		(intergated over theta) for all possible ([vr,vx],[vr',vx'])

    
    def _compute_sig_h_p(self):
        if self.debrief>1:
            print(self.prompt+'Computing SIG_H_P')

        #	Compute SIG_H_P for present velocity space grid, if it is needed and has not 
        #		already been computed with the present input parameters

        #	Compute sigma_H_P * v_v at all possible relative velocities
        _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx,self.ntheta))
        _Sig[:] = (self.Internal.v_v*sigma_el_p_h(self.Internal.v_v2*(0.5*CONST.H_MASS*self.vth**2/CONST.Q))).reshape(_Sig.shape, order='F') #NOTE Program slows significantly here

        #	Set SIG_H_P = vr' x vx_vx x Integral{v_v*sigma_H_P} over theta=0,
        #		2pi times differential velocity space element Vr'2pidVr'*dVx'

        self.Internal.SIG_H_P = np.zeros((self.nvr*self.nvx,self.nvr*self.nvx))
        self.Internal.SIG_H_P[:] = (self.Internal.Vr2pidVrdVx*self.Internal.vx_vx*(_Sig @ self.dtheta).reshape(self.Internal.Vr2pidVrdVx.shape, order='F')).reshape(self.Internal.SIG_H_P.shape, order='F')

        #	SIG_H_P is now vr' *vx_vx * sigma_H_P(v_v) * v_v (intergated over theta) 
        #		for all possible ([vr,vx],[vr',vx'])


    # --- generational ---

    def _compute_fh2_moments(self, fH2):
        if self.debrief > 1:
            print(self.prompt+'Computing vx and T moments of fH2')

        # Compute x flow velocity and temperature of molecular species
        vr2vx2_ran2 = np.zeros((self.nvr,self.nvx))
        for k in range(self.nx):
            self.H2_Moments.nH2[k] = np.sum(self.dvr_volume*(fH2[:,:,k] @ self.dvx))
            if self.H2_Moments.nH2[k] > 0:
                self.H2_Moments.VxH2[k] = self.vth*np.sum(self.dvr_volume*(fH2[:,:,k] @ (self.mesh.vx*self.dvx)))/self.H2_Moments.nH2[k]
                for i in range(self.nvr):
                    vr2vx2_ran2[i,:] = self.mesh.vr[i]**2 + (self.mesh.vx - self.H2_Moments.VxH2[k]/self.vth)**2
                self.H2_Moments.TH2[k] = (2*self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*((vr2vx2_ran2*fH2[:,:,k]) @ self.dvx)) / (3*CONST.Q*self.H2_Moments.nH2[k])


    def _compute_ni(self, nHP, ni_correct):
        if self.debrief>1:
            print(self.prompt+'Computing ni profile')
        self.Internal.ni = self.mesh.ne
        if ni_correct:
            self.Internal.ni = self.mesh.ne - nHP
        self.Internal.ni = np.maximum(self.Internal.ni, 0.01*self.mesh.ne)


    def _compute_sn(self, fSH, recomb):
        # Compute Total Atomic Hydrogen Source
        self.Internal.Sn = np.zeros((self.nvr,self.nvx,self.nx))

        # Add Recombination (optionally) and User-Supplied Hydrogen Source (velocity space distribution)
        for k in range(self.nx):
            self.Internal.Sn[:,:,k] = fSH[:,:,k]/self.vth
            if recomb:
                self.Internal.Sn[:,:,k] = self.Internal.Sn[:,:,k] + self.Internal.fi_hat[:,:,k]*self.Internal.ni[k]*self.Internal.Rec[k]

    
    def _compute_alpha_cx(self):
        #	Compute Alpha_CX for present Ti and ni, if it is needed and has not
        #		already been computed with the present parameters   
        if self.debrief > 1:
            print(self.prompt+'Computing Alpha_CX')

        if self.COLLISIONS['SIMPLE_CX']:
            #	Option (B): Use maxwellian weighted <sigma v>

            #	Charge Exchange sink rate

            self.Internal.Alpha_CX = sigmav_cx_h0(self.Internal.Ti_mu, self.Internal.ErelH_P) / self.vth
            for k in range(self.nx):
                self.Internal.Alpha_CX[:,:,k] = self.Internal.Alpha_CX[:,:,k]*self.Internal.ni[k]

        else: #NOTE Not Tested Yet
            #	Option (A): Compute SigmaV_CX from sigma directly via SIG_CX

            self.Internal.Alpha_CX = np.zeros((self.nvr,self.nvx,self.nx))
            for k in range(self.nx):
                Work = (self.Internal.fi_hat[:,:,k]*self.Internal.ni[k]).reshape((self.nvr*self.nvx), order='F')
                self.Internal.Alpha_CX[:,:,k] = (self.Internal.SIG_CX @ Work).reshape(self.Internal.Alpha_CX[:,:,k].shape, order='F')
            
            if self.Do_Alpha_CX_Test:
                Alpha_CX_Test = sigmav_cx_h0(self.Internal.Ti_mu, self.Internal.ErelH_P) / self.vth
                for k in range(self.nx):
                    Alpha_CX_Test[:,:,k] = Alpha_CX_Test[:,:,k]*self.Internal.ni[k]
                print('Compare alpha_cx and alpha_cx_test')


    def _compute_alpha_h_h2(self, fH2):
        #	Compute Alpha_H_H2 for inputted fH, if it is needed and has not
        #		already been computed with the present input parameters
        if self.debrief > 1:
            print(self.prompt+'Computing Alpha_H_H2')
        self.Internal.Alpha_H_H2 = np.zeros((self.nvr,self.nvx,self.nx))
        for k in range(self.nx):
            Work = fH2[:,:,k].reshape((self.nvr*self.nvx), order='F')
            self.Internal.Alpha_H_H2[:,:,k] = (self.Internal.SIG_H_H2 @ Work).reshape(self.Internal.Alpha_H_H2[:,:,k].shape, order='F')


    def _compute_alpha_h_p(self):
        #	Compute Alpha_H_P for present Ti and ni 
        #		if it is needed and has not already been computed with the present parameters
        if self.debrief > 1:
            print(self.prompt+'Computing Alpha_H_P')
        self.Internal.Alpha_H_P = np.zeros((self.nvr,self.nvx,self.nx))
        for k in range(self.nx):
            Work = (self.Internal.fi_hat[:,:,k]*self.Internal.ni[k]).reshape((self.nvr*self.nvx), order='F')
            self.Internal.Alpha_H_P[:,:,k] = (self.Internal.SIG_H_P @ Work).reshape(self.Internal.Alpha_H_P[:,:,k].shape, order='F')    


    # ------ Error Computation ------

    def _compute_vbar_error(self):
        if self.debrief>1:
            print(self.prompt+'Computing Vbar_Error')

        #	Test: The average speed of a non-shifted maxwellian should be 2*Vth*sqrt(Ti[x]/Tnorm)/sqrt(pi)

        vx_shift = np.zeros(self.nx)
        Tmaxwell = self.mesh.Ti
        mol = 1
        Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
        
        vbar_test = np.zeros((self.nvr,self.nvx,self.ntheta))
        vbar_error = np.zeros(self.nx)
        for m in range(self.ntheta):
            vbar_test[:,:,m] = self.Internal.vr2vx2[:,:,0]
        _vbar_test = np.zeros((self.nvr*self.nvx,self.ntheta))
        _vbar_test[:] = (self.vth*np.sqrt(vbar_test)).reshape(_vbar_test.shape, order='F')
        vbar_test = np.zeros((self.nvr,self.nvx))
        vbar_test[:] = (_vbar_test @ self.dtheta).reshape(vbar_test.shape, order='F')
        for k in range(self.nx):
            vbar = np.sum(self.dvr_volume*((vbar_test*Maxwell[:,:,k]) @ self.dvx))
            vbar_exact = 2*self.vth*np.sqrt(self.mesh.Ti[k]/self.mesh.Tnorm) / np.sqrt(np.pi)
            vbar_error[k] = abs(vbar-vbar_exact) / vbar_exact
        if self.debrief > 0:
            print(self.prompt+'Maximum Vbar error = ', sval(max(vbar_error)))


    def _compute_final_errors(self, Sion, WallH, NetHSource, Beta_CX_sum, fH, H_H2_EL, MH_H2_sum, MH_P_sum, alpha_c, Omega_H_P, Omega_H_H2, Omega_H_H, qxH_total, QH_total, debug):
        if self.debrief > 1:
            print(self.prompt+'Computing Collision Operator, Mesh, and Moment Normalized Errors')

        #	Compute Mesh Errors

        mesh_error = np.zeros((self.nvr,self.nvx,self.nx))
        max_mesh_error = 0.0
        min_mesh_error = 0.0
        mtest = 5
        moment_error = np.zeros((self.nx,mtest))
        max_moment_error = np.zeros(mtest)
        C_error = np.zeros(self.nx)
        CX_error = np.zeros(self.nx)
        H_H_error = np.zeros((self.nx, 3))
        H_H2_error = np.zeros((self.nx, 3))
        H_P_error = np.zeros((self.nx, 3))
        max_H_H_error = np.zeros(3)
        max_H_H2_error = np.zeros(3)
        max_H_P_error = np.zeros(3)

        NetHSource2 = self.Output.SourceH + self.Output.SRecomb - Sion - WallH
        for k in range(self.nx):
            C_error[k] = abs(NetHSource[k] - NetHSource2[k]) / max(abs(np.array([NetHSource[k], NetHSource2[k]])))

        #	Test conservation of particles for charge exchange operator
        if self.COLLISIONS['H_P_CX']:
            for k in range(self.nx):
                CX_A = np.sum(self.dvr_volume*((self.Internal.Alpha_CX[:,:,k]*fH[:,:,k]) @ self.dvx))
                CX_B = np.sum(self.dvr_volume*(Beta_CX_sum[:,:,k] @ self.dvx))
                CX_error[k] = np.abs(CX_A - CX_B) / np.max(np.abs(np.array([CX_A, CX_B])))

        #	Test conservation of particles, x momentum, and total energy of elastic collision operators
        for m in range(0, 3):
            for k in range(0, self.nx):
                if m < 2:
                    TfH = np.sum(self.dvr_volume*(fH[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                else:
                    TfH = np.sum(self.dvr_volume*((self.Internal.vr2vx2[:,:,k]*fH[:,:,k]) @ self.dvx))

                if self.COLLISIONS['H_H_EL']:
                    if m < 2:
                        TH_H = np.sum(self.dvr_volume*(self.Internal.MH_H_sum[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                    else:
                        TH_H = np.sum(self.dvr_volume*((self.Internal.vr2vx2[:,:,k]*self.Internal.MH_H_sum[:,:,k]) @ self.dvx))
                    H_H_error[k,m] = np.abs(TfH - TH_H) / np.max(np.abs(np.array([TfH, TH_H])))
                
                if H_H2_EL:
                    if m < 2:
                        TH_H2 = np.sum(self.dvr_volume*(MH_H2_sum[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                    else:
                        TH_H2 = np.sum(self.dvr_volume*((self.Internal.vr2vx2[:,:,k]*MH_H2_sum[:,:,k]) @ self.dvx))
                    H_H2_error[k,m] = np.abs(TfH - TH_H2) / np.max(np.abs(np.array([TfH, TH_H2])))

                if self.COLLISIONS['H_P_EL']:
                    if m < 2:
                        TH_P = np.sum(self.dvr_volume*(MH_P_sum[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                    else:
                        TH_P = np.sum(self.dvr_volume*((self.Internal.vr2vx2[:,:,k]*MH_P_sum[:,:,k]) @ self.dvx))
                    H_P_error[k,m] = np.abs(TfH - TH_P) / np.max(np.abs(np.array([TfH, TH_P])))

            max_H_H_error[m] = np.max(H_H_error[:,m])
            max_H_H2_error[m] = np.max(H_H2_error[:,m])
            max_H_P_error[m] = np.max(H_P_error[:,m])

        if self.CI_Test:
            #	Compute Momentum transfer rate via full collision integrals for charge exchange and 
            #		mixed elastic scattering.
            #		Then compute error between this and actual momentum transfer 
            #		resulting from CX and BKG (elastic) models.

            if self.COLLISIONS['H_P_CX']: # P -> H charge exchange momentum transfer via full collision integral
                print(self.prompt, 'Computing P -> H2 Charge Exchange Momentum Transfer')
                _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx,self.ntheta))
                _Sig[:] = (self.Internal.v_v*sigma_cx_h0(self.Internal.v_v2*(0.5*CONST.H_MASS*self.vth**2 / CONST.Q))).reshape(_Sig.shape, order='F')
                SIG_VX_CX = np.zeros((self.nvr*self.nvx,self.nvr*self.nvx))
                SIG_VX_CX[:] = (self.Internal.Vr2pidVrdVx*self.Internal.vx_vx*((_Sig @ self.dtheta).reshape(self.Internal.vx_vx.shape, order='F'))).reshape(SIG_VX_CX.shape, order='F')
                alpha_vx_cx = np.zeros((self.nvr,self.nvx,self.nx))

                for k in range(0, self.nx):
                    Work = (self.Internal.fi_hat[:,:,k]*self.Internal.ni[k]).reshape((self.nvr*self.nvx), order='F')
                    alpha_vx_cx[:,:,k] = (SIG_VX_CX @ Work).reshape(alpha_vx_cx[:,:,k].shape, order='F')

                RxCI_CX = np.zeros(self.nx)
                for k in range(0, self.nx):
                    RxCI_CX[k] = -(self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*((alpha_vx_cx[:,:,k]*fH[:,:,k]) @ self.dvx))

                norm = np.max(np.abs(np.array([self.Output.RxHCX, RxCI_CX])))
                CI_CX_error = np.zeros(self.nx)
                for k in range(0, self.nx):
                    CI_CX_error[k] = np.abs(self.Output.RxHCX[k] - RxCI_CX[k]) / norm

                print(self.prompt,'Maximum normalized momentum transfer error in CX collision operator: ', sval(np.max(CI_CX_error)))

            if self.COLLISIONS['H_P_EL']: # P -> H momentum transfer via full collision integral
                RxCI_P_H = np.zeros(self.nx)
                for k in range(0, self.nx):
                    RxCI_P_H[k] = -(1/2)*(self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*((self.Internal.Alpha_H_P[:,:,k]*fH[:,:,k]) @ self.dvx))

                norm = np.max(np.abs(np.array([self.Output.RxP_H, RxCI_P_H])))
                CI_P_H_error = np.zeros(self.nx)
                for k in range(0, self.nx):
                    CI_P_H_error[k] = np.abs(self.Output.RxP_H[k] - RxCI_P_H[k]) / norm 

                print(self.prompt, 'Maximum normalized momentum transfer error in P -> H elastic BKG collision operator: ', sval(np.max(CI_P_H_error)))

            if H_H2_EL: # H2 -> H momentum transfer via full collision integral
                RxCI_H2_H = np.zeros(self.nx)
                for k in range(0, self.nx):
                    RxCI_H2_H[k] = -(2/3)*(self.mu*CONST.H_MASS)*self.vth**2*np.sum(self.dvr_volume*((self.Internal.Alpha_H_H2[:,:,k]*fH[:,:,k]) @ self.dvx))
                
                norm = np.max(np.abs(np.array([self.Output.RxH2_H, RxCI_H2_H])))
                CI_H2_H_error = np.zeros(self.nx)
                for k in range(0, self.nx):
                    CI_H2_H_error[k] = np.abs(self.Output.RxH2_H[k] - RxCI_H2_H[k])/norm
                
                print(self.prompt, 'Maximum normalized momentum transfer error in H2 -> H elastic BKG collision operator: ', sval(np.max(CI_H2_H_error)))

            if self.COLLISIONS['H_H_EL']: # H -> H perp/parallel energy transfer via full collision integral
                Epara_Perp_CI = np.zeros(self.nx)
                for k in range(0, self.nx):
                    Work = fH[:,:,k].reshape((self.nvr*self.nvx), order='F')
                    Alpha_H_H = (self.Internal.SIG_H_H @ Work).reshape((self.nvr,self.nvx), order='F')
                    Epara_Perp_CI[k] = 0.5*(self.mu*CONST.H_MASS)*self.vth**3*np.sum(self.dvr_volume*((Alpha_H_H*fH[:,:,k]) @ self.dvx)) 
                
                norm = np.max(np.abs(np.array([self.Output.Epara_PerpH_H, Epara_Perp_CI])))
                CI_H_H_error = np.zeros(self.nx)
                for k in range(0, self.nx):
                    CI_H_H_error[k] = np.abs(self.Output.Epara_PerpH_H[k] - Epara_Perp_CI[k]) / norm 
                
                print(self.prompt, 'Maximum normalized perp/parallel energy transfer error in H -> H elastic BKG collision operator: ', sval(np.max(CI_H_H_error)))

        #	Mesh Point Error based on fH satisfying Boltzmann equation

        T1 = np.zeros((self.nvr,self.nvx,self.nx))
        T2 = np.zeros((self.nvr,self.nvx,self.nx))
        T3 = np.zeros((self.nvr,self.nvx,self.nx))
        T4 = np.zeros((self.nvr,self.nvx,self.nx))
        T5 = np.zeros((self.nvr,self.nvx,self.nx))
        for k in range(0, self.nx-1):
            for j in range(0, self.nvx):
                T1[:,j,k] = 2*self.mesh.vx[j]*(fH[:,j,k+1] - fH[:,j,k]) / (self.mesh.x[k+1] - self.mesh.x[k]) 
            T2[:,:,k] = (self.Internal.Sn[:,:,k+1] + self.Internal.Sn[:,:,k])
            T3[:,:,k] = Beta_CX_sum[:,:,k+1] + Beta_CX_sum[:,:,k]
            T4[:,:,k] = alpha_c[:,:,k+1]*fH[:,:,k+1] + alpha_c[:,:,k]*fH[:,:,k]
            T5[:,:,k] = Omega_H_P[k+1]*MH_P_sum[:,:,k+1] + Omega_H_H2[k+1]*MH_H2_sum[:,:,k+1] + Omega_H_H[k+1]*self.Internal.MH_H_sum[:,:,k+1] + \
                    Omega_H_P[k]*MH_P_sum[:,:,k] + Omega_H_H2[k]*MH_H2_sum[:,:,k] + Omega_H_H[k]*self.Internal.MH_H_sum[:,:,k]
            mesh_error[:,:,k] = np.abs(T1[:,:,k] - T2[:,:,k] - T3[:,:,k] + T4[:,:,k] - T5[:,:,k]) / \
                                np.max(np.abs(np.array([T1[:,:,k], T2[:,:,k], T3[:,:,k], T4[:,:,k], T5[:,:,k]])))
        ave_mesh_error = np.sum(mesh_error) / np.size(mesh_error)
        max_mesh_error = np.max(mesh_error)
        min_mesh_error = np.min(mesh_error[:,:,0:self.nx-1])

        #	Moment Error
        for m in range(0, mtest):
            for k in range(0, self.nx-1):
                MT1 = np.sum(self.dvr_volume*(T1[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                MT2 = np.sum(self.dvr_volume*(T2[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                MT3 = np.sum(self.dvr_volume*(T3[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                MT4 = np.sum(self.dvr_volume*(T4[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                MT5 = np.sum(self.dvr_volume*(T5[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                #NOTE This is correct for the original code, but is it correct mathematically?
                moment_error[k,m] = np.abs(MT1 - MT2 - MT3 + MT4 - MT5) / np.max(np.abs(np.array([MT1, MT2, MT3, MT4, MT5])))
            max_moment_error[m] = np.max(moment_error[:,m])

        #	Compute error in qxH_total

        #		qxH_total2 total neutral heat flux profile (watts m^-2)
        #			This is the total heat flux transported by the neutrals
        #			computed in a different way from:

        #			qxH_total2(k)=Vth**3*total(Vr2pidVr*((vr2vx2(*,*,k)*fH(*,*,k))#(Vx*dVx)))*0.5*(mu*mH)

        #			This should agree with qxH_total if the definitions of nH, pH, piH_xx,
        #			TH, VxH, and qxH are coded correctly.
        qxH_total2 = np.zeros(self.nx)
        for k in range(0, self.nx):
            qxH_total2[k] = 0.5*(self.mu*CONST.H_MASS)*self.vth**3*np.sum(self.dvr_volume*((self.Internal.vr2vx2[:,:,k]*fH[:,:,k]) @ (self.mesh.vx*self.dvx)))
        qxH_total_error = np.abs(qxH_total - qxH_total2) / np.max(np.abs(np.array([qxH_total, qxH_total2])))

        #	Compute error in QH_total
        Q1 = np.zeros(self.nx)
        Q2 = np.zeros(self.nx)
        QH_total_error = np.zeros(self.nx)
        for k in range(0, self.nx-1):
            Q1[k] = (qxH_total[k+1] - qxH_total[k]) / (self.mesh.x[k+1] - self.mesh.x[k])
            Q2[k] = 0.5*(QH_total[k+1] + QH_total[k])
        QH_total_error = np.abs(Q1 - Q2) / np.max(np.abs(np.array([Q1, Q2])))

        if self.debrief > 0:
            print(self.prompt+'Maximum particle convervation error of total collision operator: '+sval(max(C_error)))
            print(self.prompt+'Maximum H_P_CX  particle convervation error: '+sval(max(CX_error)))
            print(self.prompt+'Maximum H_H_EL  particle conservation error: '+sval(max_H_H_error[0]))
            print(self.prompt+'Maximum H_H_EL  x-momentum conservation error: '+sval(max_H_H_error[1]))
            print(self.prompt+'Maximum H_H_EL  total energy conservation error: '+sval(max_H_H_error[2]))
            print(self.prompt+'Maximum H_H2_EL particle conservation error: '+sval(max_H_H2_error[0]))
            print(self.prompt+'Maximum H_P_EL  particle conservation error: '+sval(max_H_P_error[0]))
            print(self.prompt+'Average mesh_error = '+str(ave_mesh_error))
            print(self.prompt+'Maximum mesh_error = '+str(max_mesh_error))
            for m in range(5):
                print(self.prompt+'Maximum fH vx^'+sval(m)+' moment error: '+sval(max_moment_error[m]))
            print(self.prompt+'Maximum qxH_total error = '+str(max(qxH_total_error)))
            print(self.prompt+'Maximum QH_total error = '+str(max(QH_total_error)))
            if debug > 0:
                input()