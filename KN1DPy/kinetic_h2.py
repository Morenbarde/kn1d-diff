from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import copy

from .make_dvr_dvx import VSpace_Differentials
from .create_shifted_maxwellian import create_shifted_maxwellian
from .kinetic_mesh import KineticMesh

from .sigma.sigmav_ion_hh import sigmav_ion_hh
from .sigma.sigmav_h1s_h1s_hh import sigmav_h1s_h1s_hh
from .sigma.sigmav_h1s_h2s_hh import sigmav_h1s_h2s_hh
from .sigma.sigmav_p_h1s_hh import sigmav_p_h1s_hh
from .sigma.sigmav_h2p_h2s_hh import sigmav_h2p_h2s_hh
from .sigma.sigmav_h1s_hn3_hh import sigmav_h1s_hn3_hh
from .sigma.sigmav_p_h1s_hp import sigmav_p_h1s_hp
from .sigma.sigmav_p_hn2_hp import sigmav_p_hn2_hp
from .sigma.sigmav_p_p_hp import sigmav_p_p_hp
from .sigma.sigmav_h1s_hn_hp import sigmav_h1s_hn_hp
from .sigma.sigma_cx_hh import sigma_cx_hh
from .sigma.sigma_el_h_hh import sigma_el_h_hh
from .sigma.sigma_el_p_hh import sigma_el_p_hh
from .sigma.sigma_el_hh_hh import sigma_el_hh_hh
from .sigma.sigmav_cx_hh import sigmav_cx_hh

from .utils import sval, get_config, path_interp_2d

from .common.Kinetic_H2 import *
from .common import constants as CONST


# Dataclasses for use in kinetic_h

@dataclass
class KH2Collisions():
    '''
    Collision settings for Kinetic H2 procedure
    '''
    H2_H_EL: bool = False
    H2_H2_EL: bool = False
    H2_P_EL: bool = False
    H2_P_CX: bool = False
    Simple_CX: bool = False


class KineticH2():

    # Internal Debug switches
    CI_Test = True
    Do_Alpha_CX_Test = False

    # Internal Tolerances 
    DeltaVx_tol = .01
    Wpp_tol = .001

    # Theta-prime Coordinate
    ntheta = 5      # use 5 theta mesh points for theta integration
    dtheta = np.ones(ntheta)/ntheta
    cos_theta = np.cos(np.pi*(np.arange(ntheta) + 0.5) / ntheta)

    # Internal Print Formatting
    prompt = 'Kinetic_H2 => '



    def __init__(self, mesh: KineticMesh, mu: int, vxi: NDArray, fH2BC: NDArray, GammaxH2BC: float, NuLoss: NDArray):

        # Configuration Options
        self.config = get_config()

        col = self.config['collisions']
        self.COLLISIONS = KH2Collisions(col['H2_H_EL'], col['H2_H2_EL'], col['H2_P_EL'], col['H2_P_CX'], col['SIMPLE_CX'])

        # Main attributes
        self.mesh = mesh
        self.mu = mu
        self.vxi = vxi
        self.fH2BC = fH2BC
        self.GammaxH2BC = GammaxH2BC
        self.NuLoss = NuLoss

        # Shorthand sizes for main mesh variables
        self.nvr = mesh.vr.size
        self.nvx = self.mesh.vx.size
        self.nx = self.mesh.x.size

        self.vx_neg = np.nonzero(self.mesh.vx < 0)[0]
        self.vx_pos = np.nonzero(self.mesh.vx > 0)[0]
        self.vx_zero = np.nonzero(self.mesh.vx == 0)[0]

        # Variables for internal use
        self.fH2BC_input = np.zeros(self.fH2BC.shape)
        self.fH2BC_input[:,self.vx_pos] = self.fH2BC[:,self.vx_pos]

        self.vth = np.sqrt(2 * CONST.Q * self.mesh.Tnorm / (self.mu * CONST.H_MASS))
        # Vr^2-2*Vx^2
        self.vr2_2vx2_2D = np.asarray([(vr**2) - 2*(self.mesh.vx**2) for vr in self.mesh.vr])
        # Differential Values
        differentials = VSpace_Differentials(self.mesh.vr, self.mesh.vx)
        self.dvr_vol = differentials.dvr_vol
        self.dvr_vol_h_order = differentials.dvr_vol_h_order
        self.dvx = differentials.dvx
        # Determine Energy Space Differentials 
        self.Eaxis = (self.vth**2)*0.5*self.mu*CONST.H_MASS*(self.mesh.vr**2)/CONST.Q
        _Eaxis = np.append(self.Eaxis, 2*self.Eaxis[self.nvr - 1] - self.Eaxis[self.nvr - 2])
        Eaxis_mid = np.append(0.0, 0.5*( _Eaxis + np.roll(_Eaxis, -1) ))
        self.dEaxis = np.roll(Eaxis_mid, -1) - Eaxis_mid
        self.dEaxis = self.dEaxis[0:self.nvr]


        # Common Blocks
        self.Input = Kinetic_H2_Input()
        self.Internal = Kinetic_H2_Internal()
        self.Output = Kinetic_H2_Output(self.nx)
        self.H_Moments = Kinetic_H2_H_Moments()
        self.Errors = Kinetic_H2_Errors()


        self._test_init_parameters()

        return


    def run_procedure(self, fH: NDArray = None, SH2: NDArray = None, fH2: NDArray = None, nHP: NDArray = None, THP: NDArray = None, 
                truncate: float = 1.0e-4, max_gen: int = 50,  compute_h_source: bool = False, sawada: bool = True, 
                ni_correct: bool = False, compute_errors: bool = False, plot: int = 0, debug: int = 0, debrief: int = 0, pause = False):
        '''
        Solves a 1-D spatial, 2-D velocity kinetic neutral transport 
        problem for molecular hydrogen or deuterium (H2)

        Parameters
        ----------
            mesh : KineticMesh
                Mesh data for h2 kinetic procedure, must be of type 'h2'
                Includes coordinate data and temperature/density profiles
            mu : int
                1=hydrogen, 2=deuterium
            vxi : ndarray
                flow speed profile (m/s)
            fH2BC : ndarray
                2D array, input boundary condition. Specifies shape of molecule velocity distribution (fH2) at x=0
            GammaxH2BC : float
                Desired neutral molecule flux density at x=0 (m^-2 s^-1)
            NuLoss : ndarray
                Characteristic molecular ion loss frequency profile (1/s)
            fH : ndarray, default=None
                3D array, atomic distribution function. If None, H-H2 collisions are not computed
            SH2 : ndarray, defualt=None
                Source profile of wall-temperature H2 molecules (m^-3 s^-1). If None, zero array is used
            fH2 : ndarray, defualt=None
                3D array, molecular distribution function. If None, zero array is used
            nHP : ndarray, defualt=None
                Molecular ion density profile (m^-3). If None, zero array is used
            THP : ndarray, defualt=None
                Molecular ion temperature profile (m^-3). If None, array of 3.0 used
            KH2 : Kinetic_H2_Common, default=None
                Common blocks used to pass data.
                NOTE Consider changing this, program will currently fail if not set
            truncate : float, default=1.0e-4
                Convergence threshold for generations
            max_gen : int, default=50
                Max number of generations
            compute_h_source : bool, default=False
                If true, compute fSH, SH, SP, and SHP
            sawada : bool, default=False
                If false, disable Sawada correction
            ni_correct : bool, default=False
                If true, Corrects hydrogen ion density according to quasineutrality: ni=ne-nHp
            compute_errors : bool, default=False
                If true, compute error estimates
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
            pause : bool, default=false
                If true, pause between plots

        Returns
        -------
            fH2 : ndarray
                3D array, molecular distribution function.
            nHP : ndarray, defualt=None
                Molecular ion density profile (m^-3).
            THP : ndarray, defualt=None
                Molecular ion temperature profile (m^-3).
            nH2 : ndarray
                Molecular density profile (m^-3)
            GammaxH2 : ndarray
                Neutral flux profile (m^-2 s^-1)
            VxH2 : ndarray
                Neutral velocity profile (m s^-1)
            pH2 : ndarray
                Neutral pressure (eV m^-2)
            TH2 : ndarray
                Neutral temperature profile (eV)
            qxH2 : ndarray
                Neutral random heat flux profile (watts m^-2)
            qxH2_total : ndarray
                Total neutral heat flux profile (watts m^-2)
            Sloss : ndarray
                H2 loss rate from ionization and dissociation (SH2) (m^-3 s^-1)
            QH2 : ndarray
                Rate of net thermal energy transfer into neutral molecules (watts m^-3)
            RxH2 : ndarray
                Rate of x momentum transfer to neutral molecules (N m^-2)
            QH2_total : ndarray
                Net rate of total energy transfer into neutral molecules (watts m^-3)
            AlbedoH2 : float
                Ratio of molceular particle flux with Vx < 0 divided by particle flux with Vx > 0 at x=0
            WallH2 : ndarray
                Molecular sink rate and source rate from interation with 'side walls' (m^-3 s^-1)
            fSH : ndarray
                3D array, H source velocity distribution
            SH : ndarray
                Atomic neutral source profile (m^-3 s^-1)
            SP : ndarray
                Proton source profile (m^-3 s^-1)
            SHP : ndarray
                Molecular ion source profile (m^-3 s^-1)
            NuE : ndarray
                Energy equilibration frequency of molecular ion with bulk plasma (1/s)
            NuDis : ndarray
                Molecular ion dissociation frequency (1/s)
            ESH : ndarray
                Normalized H source energy distribution
            Eaxis : ndarray
                Energy coordinate for ESH (eV)

        Notes
        -------
            This subroutine is part of the "KN1D" atomic and molecular neutal transport code.

            This subroutine solves a 1-D spatial, 2-D velocity kinetic neutral transport
            problem for molecular hydrogen or deuterium (H2) by computing successive generations of
            charge exchange and elastic scattered neutrals. The routine handles electron-impact
            ionization and dissociation, molecular ion charge exchange, and elastic
            collisions with hydrogenic ions, neutral atoms, and molecules.

            The positive vx half of the atomic neutral distribution function is inputted at x(0)
            (with arbitrary normalization). The desired flux on molecules entering the slab geometry at
            x(0) is specified. Background profiles of plasma ions, (e.g., Ti(x), Te(x), n(x), vxi(x),...)
            and atomic distribution function (fH) is inputted. (fH can be computed by procedure 
            "Kinetic_H.pro".) Optionally, a molecular source profile (SH2(x)) is also inputted. 
            The code returns the molecular hydrogen distribution function, fH2(vr,vx,x) for all 
            vx, vr, and x of the specified vr,vx,x grid. The atomic (H) and ionic (P) hydrogen 
            source profiles and the atomic source velocity distribution functions 
            resulting from Franck-Condon reaction product energies of H are also returned.

            Since the problem involves only the x spatial dimension, all distribution functions
            are assumed to have rotational symmetry about the vx axis. Consequently, the distributions
            only depend on x, vx and vr where vr =sqrt(vy^2+vz^2)

            History:
                B. LaBombard   First coding based on Kinetic_Neutrals.pro 22-Dec-2000

                For more information, see write-up: "A 1-D Space, 2-D Velocity, Kinetic
                Neutral Transport Algorithm for Hydrogen Molecules in an Ionizing Plasma", B. LaBombard

            Variable names contain characters to help designate species -
            atomic neutral (H), molecular neutral (H2), molecular ion (HP), proton (i) or (P)
        '''


        # Kinetic_H2_input common - will change later 
        vx_s = self.Input.vx_s
        vr_s = self.Input.vr_s
        x_s = self.Input.x_s
        Tnorm_s = self.Input.Tnorm_s
        mu_s = self.Input.mu_s
        Ti_s = self.Input.Ti_s
        Te_s = self.Input.Te_s
        n_s = self.Input.n_s
        vxi_s = self.Input.vxi_s
        fH_s = self.Input.fH_s
        fH2_s = self.Input.fH2_s

        nHP_s = self.Input.nHP_s
        THP_s = self.Input.THP_s
        Simple_CX_s = self.Input.Simple_CX_s
        ni_correct_s = self.Input.ni_correct_s

        # kinetic_h2_internal common block - will change later 
        vr2vx2 = self.Internal.vr2vx2
        vr2vx_vxi2 = self.Internal.vr2vx_vxi2
        fw_hat = self.Internal.fw_hat
        fi_hat = self.Internal.fi_hat
        fHp_hat = self.Internal.fHp_hat
        EH2_P = self.Internal.EH2_P
        sigv = self.Internal.sigv
        Alpha_Loss = self.Internal.Alpha_Loss
        v_v2 = self.Internal.v_v2
        v_v = self.Internal.v_v
        vr2_vx2 = self.Internal.vr2_vx2
        vx_vx = self.Internal.vx_vx

        Vr2pidVrdVx = self.Internal.Vr2pidVrdVx
        SIG_CX = self.Internal.SIG_CX
        SIG_H2_H2 = self.Internal.SIG_H2_H2
        SIG_H2_H = self.Internal.SIG_H2_H
        SIG_H2_P = self.Internal.SIG_H2_P
        Alpha_CX = self.Internal.Alpha_CX
        Alpha_H2_H = self.Internal.Alpha_H2_H
        MH2_H2_sum = self.Internal.MH2_H2_sum
        Delta_nH2s = self.Internal.Delta_nH2s

        # Override settings for debug
        if debug > 0: 
            # plot = np.maximum(plot, 1)
            debrief = np.maximum(debrief, 1)
            pause = True   # not sure if this is the correct translation 


        # --- Initialize Inputs ---
        
        if fH is None:
            fH = np.arange((self.nvr, self.nvx, self.nx))
        if fH2 is None:
            fH2 = np.zeros((self.nvr,self.nvx,self.nx))
        if SH2 is None:
            SH2 = np.zeros(self.nx)
        if nHP is None:
            nHP = np.zeros(self.nx)
        if THP is None:
            THP = np.full(self.nx, 3.0)
        self._test_input_parameters(fH, fH2, SH2, nHP, THP)

        # if fh is zero, then turn off elastic H2 <-> H collisions
        self.COLLISIONS.H2_H_EL = self.config['collisions']['H2_H_EL']
        if np.sum(fH) <= 0.0:
            self.COLLISIONS.H2_H_EL = False
    
        # Reset H Moments
        self.H_Moments.nH = np.zeros(self.nx)
        self.H_Moments.VxH = np.zeros(self.nx)
        self.H_Moments.TH = np.zeros(self.nx)

        # Scale input molecular distribution function to agree with desired flux
        gamma_input = 1.0
        if abs(self.GammaxH2BC) > 0:
            gamma_input = self.vth*np.sum(self.dvr_vol*(self.fH2BC_input @ (self.mesh.vx*self.dvx)))
        ratio = abs(self.GammaxH2BC)/gamma_input
        fH2BC_input = self.fH2BC_input*ratio
        if abs(ratio - 1) > 0.01*truncate:
            self.fH2BC = fH2BC_input
        fH2[:,self.vx_pos,0] = fH2BC_input[:,self.vx_pos]

        
        # Set flags to make use of previously computed local parameters 
        New_Grid = 1#
        if vx_s is not None:
            test = 0 
            ii = np.argwhere(vx_s != self.mesh.vx) ; test = test + np.size(ii)
            ii = np.argwhere(vr_s != self.mesh.vr) ; test = test + np.size(ii)
            ii = np.argwhere(x_s != self.mesh.x) ; test = test + np.size(ii)
            ii = np.argwhere(Tnorm_s != self.mesh.Tnorm) ; test = test + np.size(ii)
            ii = np.argwhere(mu_s != self.mu) ; test = test + np.size(ii)
            if test <= 0:
                New_Grid = 0
        New_Protons = 1#
        if Ti_s is not None:
            test = 0 
            ii = np.argwhere(Ti_s != self.mesh.Ti)
            test = test + np.size(ii)
            ii = np.argwhere(n_s != self.mesh.ne)
            test = test + np.size(ii)
            ii = np.argwhere(vxi_s != self.vxi)
            test = test + np.size(ii)
            if test <= 0:
                New_Protons = 0
        New_Electrons = 1#
        if Te_s is not None:
            test = 0 
            ii = np.argwhere(Te_s != self.mesh.Te) ; test = test + np.size(ii)
            ii = np.argwhere(n_s != self.mesh.ne) ; test = test + np.size(ii)
            if test <= 0:
                New_Electrons = 0
        New_fH = 1
        if fH_s is not None:
            ii = np.argwhere(fH_s != fH)
            if np.size(ii) <= 0:
                New_fH = 0
        New_Simple_CX = 1#
        if Simple_CX_s is not None:
            ii = np.argwhere(Simple_CX_s != self.COLLISIONS.Simple_CX)
            if np.size(ii) <= 0:
                New_Simple_CX = 0
        New_H2_Seed=1
        if fH2_s is not None:
            ii = np.argwhere(fH2_s != fH2)
            if np.size(ii) <= 0:
                New_H2_Seed = 0
        New_HP_Seed=1
        if nHP_s is not None:
            test = 0
            ii = np.argwhere(nHP_s != nHP)
            test = test + np.size(ii)
            ii = np.argwhere(THP_s != THP)
            test = test + np.size(ii)
            if test <= 0:
                New_HP_Seed = 0
        New_ni_correct = True
        if (ni_correct_s is not None) and (ni_correct_s != ni_correct):
            New_ni_correct = False 

        Do_sigv = New_Grid | New_Electrons
        Do_fH_moments = (New_Grid | New_fH) & (np.sum(fH) > 0.0)
        Do_Alpha_CX =   (New_Grid | (Alpha_CX is None) | New_HP_Seed | New_Simple_CX) & self.COLLISIONS.H2_P_CX
        # Do_Alpha_CX is updated in fH2_iteration loop
        Do_SIG_CX =     (New_Grid | (SIG_CX is None) | New_Simple_CX) & (self.COLLISIONS.Simple_CX == 0) & Do_Alpha_CX
        Do_Alpha_H2_H = (New_Grid | (Alpha_H2_H is None) | New_fH) & self.COLLISIONS.H2_H_EL
        Do_SIG_H2_H =   (New_Grid | (SIG_H2_H is None)) & Do_Alpha_H2_H
        Do_SIG_H2_H2 =  (New_Grid | (SIG_H2_H2 is None)) & self.COLLISIONS.H2_H2_EL
        Do_Alpha_H2_P = (New_Grid | (not 'Alpha_H2_P' in locals()) | New_Protons | New_ni_correct) & self.COLLISIONS.H2_P_EL #COME BACK TO THIS Alpha_H2_p is not defined!!!!!
        # Do_Alpha_H2_P is updated in fH2_iteration loop
        Do_SIG_H2_P =   (New_Grid | (SIG_H2_P is None)) & Do_Alpha_H2_P
        Do_v_v2 =      (New_Grid or (v_v2 is None)) & (self.CI_Test | Do_SIG_CX | Do_SIG_H2_H | Do_SIG_H2_H2 | Do_SIG_H2_P)

        if debug > 0:
            print("Kinetic H2 Settings")
            print("H2_H2_EL", self.COLLISIONS.H2_H2_EL)
            print("H2_P_EL", self.COLLISIONS.H2_P_EL)
            print("H2_H_EL", self.COLLISIONS.H2_H_EL)
            print("H2_P_CX", self.COLLISIONS.H2_P_CX)
            print("New_Grid", New_Grid)
            print("New_Protons", New_Protons)
            print("New_Electrons", New_Electrons)
            print("New_fH", New_fH)
            print("New_Simple_CX", New_Simple_CX)
            print("New_H2_Seed", New_H2_Seed)
            print("New_HP_Seed", New_HP_Seed)
            print("New_ni_correct", New_ni_correct)
            print("Do_sigv", Do_sigv)
            print("Do_fH_moments", Do_fH_moments)
            print("Do_Alpha_CX", Do_Alpha_CX)
            print("Do_SIG_CX", Do_SIG_CX)
            print("Simple_CX", self.COLLISIONS.Simple_CX)
            print("Do_Alpha_H2_H", Do_Alpha_H2_H)
            print("Do_SIG_H2_H", Do_SIG_H2_H)
            print("Do_SIG_H2_H2", Do_SIG_H2_H2)
            print("Do_Alpha_H2_P", Do_Alpha_H2_P)
            print("Do_SIG_H2_P", Do_SIG_H2_P)
            print("Do_v_v2", Do_v_v2)
            input()

        nH = np.zeros(self.nx)
        VxH = np.zeros(self.nx)
        TH = np.zeros(self.nx) + 1.0

        if Do_fH_moments:
            if debrief > 1:
                print(self.prompt, 'Computing vx and T moments of fH')

            # Compute x flow velocity and temperature of atomic species
            vr2vx2_ran2 = np.zeros((self.nvr,self.nvx))
            for k in range(0, self.nx):
                nH[k] = np.sum(self.dvr_vol*(fH[:,:,k] @ self.dvx))
                if nH[k] > 0:
                    VxH[k] = self.vth*np.sum(self.dvr_vol*(fH[:,:,k] @ (self.mesh.vx*self.dvx)))/nH[k]
                    for i in range(0, self.nvr):
                        vr2vx2_ran2[i,:] = self.mesh.vr[i]**2 + (self.mesh.vx - VxH[k]/self.vth)**2
                    TH[k] = (self.mu*CONST.H_MASS)*(self.vth**2)*np.sum((self.dvr_vol*((vr2vx2_ran2*fH[:,:,k]) @ self.dvx))) / (3*CONST.Q*nH[k])
        
        if New_Grid:
            if debrief > 1:
                print(self.prompt, 'Computing vr2vx2, vr2vx_vxi2, EH2_P')
        
            # Magnitude of total normalized v^2 at each mesh point
            vr2vx2 = np.zeros((self.nvr,self.nvx,self.nx))
            for i in range(0, self.nvr):
                for k in range(0, self.nx):
                    vr2vx2[i,:,k] = self.mesh.vr[i]**2 + self.mesh.vx**2

            # Magnitude of total normalized (v-vxi)^2 at each mesh point
            vr2vx_vxi2 = np.zeros((self.nvr,self.nvx,self.nx))
            for i in range(0, self.nvr):
                for k in range(0, self.nx):
                    vr2vx_vxi2[i,:,k] = self.mesh.vr[i]**2 + (self.mesh.vx - self.vxi[k]/self.vth)**2

            # Molecular hydrogen ion energy in local rest frame of plasma at each mesh point
            EH2_P = CONST.H_MASS*vr2vx_vxi2*(self.vth**2)/ CONST.Q
            EH2_P = np.maximum(EH2_P, 0.1)      # sigmav_cx does not handle neutral energies below 0.1 eV
            EH2_P = np.minimum(EH2_P, 2.0e4)    # sigmav_cx does not handle neutral energies above 20 keV

            # Compute Maxwellian H2 distribution at the wall temperature
            fw_hat = np.zeros((self.nvr,self.nvx))
            
            # note: Molecular ions have 'normalizing temperature' of 2 Tnorm, i.e., in order to
            # achieve the same thermal velocity^2, a molecular ion distribution has to have twice the temperature 
            # as an atomic ion distribution

            if (np.sum(SH2) > 0) | (np.sum(self.mesh.PipeDia) > 0): # come back and double check if it should be a bitwise or logical opperator 
                if debrief > 1:
                    print(self.prompt, 'Computing fw_hat')
                vx_shift = np.array([0.0])
                Tmaxwell = np.array([CONST.TWALL])
                mol = 2
                _maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
                fw_hat = _maxwell[:,:,0]

        if New_Protons:
            # Compute fi_hat 
            if debrief > 1:
                print(self.prompt, 'Computing fi_Hat')
            vx_shift = self.vxi
            Tmaxwell = self.mesh.Ti
            mol = 1
            Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
            fi_hat = copy.copy(Maxwell)

        if Do_sigv:
            if debrief > 1:
                print(self.prompt, 'Computing sigv')

            # Compute sigmav rates for each reaction and optionally apply
            # CR model corrections of Sawada

            sigv = np.zeros((self.nx,11))

            # Reaction R1:  e + H2 -> e + H2(+) + e 
            sigv[:,1] = sigmav_ion_hh(self.mesh.Te)
            if sawada:
                sigv[:,1] = sigv[:,1] * 3.7 / 2.0

            # Reaction R2:  e + H2 -> H(1s) + H(1s)
            sigv[:,2] = sigmav_h1s_h1s_hh(self.mesh.Te)
            if sawada:
                # Construct Table 
                Te_table = np.log([5,20,100])
                Ne_table = np.log([1e14,1e17,1e18,1e19,1e20,1e21,1e22])
                fctr_table = np.zeros((7, 3))
                fctr_table[:,0] = np.array([2.2, 2.2, 2.1, 1.9, 1.2,  1.1,  1.05]) / 5.3
                fctr_table[:,1] = np.array([5.1, 5.1, 4.3, 3.1, 1.5,  1.25, 1.25]) / 10.05
                fctr_table[:,2] = np.array([1.3, 1.3, 1.1, 0.8, 0.38, 0.24, 0.22]) / 2.1
                _Te = self.mesh.Te
                _Te = np.maximum(_Te, 5)
                _Te = np.minimum(_Te, 100)
                _n = np.maximum(self.mesh.ne, 1e14)
                _n = np.minimum(self.mesh.ne, 1e22)
                fctr = path_interp_2d(fctr_table, Ne_table, Te_table, np.log(_n), np.log(_Te))
                sigv[:,2] = (1.0 + fctr)*sigv[:,2]
            
            # Reaction R3:  e + H2 -> e + H(1s) + H*(2s)
            sigv[:,3] = sigmav_h1s_h2s_hh(self.mesh.Te)

            # Reaction R4:  e + H2 -> e + p + H(1s)
            sigv[:,4] = sigmav_p_h1s_hh(self.mesh.Te)
            if sawada:
                sigv[:,4] = sigv[:,4]*1.0/0.6

            # Reaction R5:  e + H2 -> e + H*(2p) + H*(2s)
            sigv[:,5] = sigmav_h2p_h2s_hh(self.mesh.Te)

            # Reaction R6:  e + H2 -> e + H(1s) + H*(n=3)
            sigv[:,6] = sigmav_h1s_hn3_hh(self.mesh.Te)

            # Reaction R7:  e + H2(+) -> e + p + H(1s)
            sigv[:,7] = sigmav_p_h1s_hp(self.mesh.Te)

            # Reaction R8:  e + H2(+) -> e + p + H*(n=2)
            sigv[:,8] = sigmav_p_hn2_hp(self.mesh.Te)

            # Reaction R9:  e + H2(+) -> e + p + p + e
            sigv[:,9] = sigmav_p_p_hp(self.mesh.Te)

            # Reaction R10:  e + H2(+) -> e + H(1s) + H*(n>=2)
            sigv[:,10] = sigmav_h1s_hn_hp(self.mesh.Te)
            
            # Total H2 destruction rate (normalized by vth) = sum of reactions 1-6
            Alpha_Loss = np.zeros(self.nx)
            Alpha_Loss[:] = self.mesh.ne*np.sum(sigv[:,1:7], axis = 1) / self.vth

        # Set up arrays for charge exchange and elastic collision computations, if needed 
        if Do_v_v2 == 1:
            if debrief > 1:
                print(self.prompt, 'Computing v_v2, v_v, vr2_vx2, and vx_vx')
            # v_v2=(v-v_prime)^2 at each double velocity space mesh point, including theta angle
            v_v2 = np.zeros((self.nvr,self.nvx,self.nvr,self.nvx,self.ntheta))

            # vr2_vx2=(vr2 + vr2_prime - 2*vr*vr_prime*cos(theta) - 2*(vx-vx_prime)^2
            # at each double velocity space mesh point, including theta angle
            vr2_vx2 = np.zeros((self.nvr,self.nvx,self.nvr,self.nvx,self.ntheta))
            for m in range(0, self.ntheta): # double check this nested for loop 
                for l in range(0, self.nvx):
                    for k in range(0, self.nvr):
                        for i in range(0, self.nvr):
                            v_starter = self.mesh.vr[i]**2 + self.mesh.vr[k]**2 - 2*self.mesh.vr[i]*self.mesh.vr[k]*self.cos_theta[m]
                            v_v2[i,:,k,l,m] = v_starter + (self.mesh.vx[:] - self.mesh.vx[l])**2  # not super confident 
                            vr2_vx2[i,:,k,l,m] = v_starter - 2*(self.mesh.vx[:] - self.mesh.vx[l])**2
            # v_v=|v-v_prime| at each double velocity space mesh point, including theta angle
            v_v = np.sqrt(v_v2)

            # vx_vx=(vx-vx_prime) at each double velocity space mesh point
            vx_vx = np.zeros((self.nvr,self.nvx,self.nvr,self.nvx))
            for j in range(0,self.nvx):
                for l in range(0, self.nvx):
                    vx_vx[:,j,:,l] = self.mesh.vx[j] - self.mesh.vx[l]

            # Set Vr'2pidVr'*dVx' for each double velocity space mesh point
            Vr2pidVrdVx = np.zeros((self.nvr,self.nvx,self.nvr,self.nvx))
            for k in range(0, self.nvr):
                Vr2pidVrdVx[:,:,k,:] = self.dvr_vol[k]
            for l in range(0, self.nvx):
                Vr2pidVrdVx[:,:,:,l] = Vr2pidVrdVx[:,:,:,l]*self.dvx[l]
        
        if self.COLLISIONS.Simple_CX == 0 and Do_SIG_CX == 1: #NOTE Not Tested Yet
            if debrief > 1:
                print(self.prompt, 'Computing SIG_CX')
            #  Option (A) was selected: Compute SigmaV_CX from sigma directly.
            # In preparation, compute SIG_CX for present velocity space grid, if it has not 
            # already been computed with the present input parameters

            # compute SIGMA_CX * v_v at all possible relative velocities
            _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx, self.ntheta))
            _Sig[:] = (v_v*sigma_cx_hh(v_v2*(CONST.H_MASS*(self.vth**2)/CONST.Q))).reshape(_Sig.shape, order='F')

            # Set SIG_CX = vr' x Integral{v_v*sigma_cx} over theta=0,2pi times differential velocity space element Vr'2pidVr'*dVx'
            SIG_CX = np.zeros((self.nvr*self.nvx, self.nvr*self.nvx))
            SIG_CX[:] = (Vr2pidVrdVx*((_Sig @ self.dtheta).reshape(Vr2pidVrdVx.shape, order='F'))).reshape(SIG_CX.shape, order='F') 

            # SIG_CX is now vr' * sigma_cx(v_v) * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])

        if Do_SIG_H2_H == 1: #NOTE Not Tested Yet
            if debrief > 1:
                print(self.prompt, 'Computing SIG_H2_P')
            # Compute SIG_H2_P for present velocity space grid, if it is needed and has not 
            # already been computed with the present input parameters

            # Compute sigma_H2_H * v_v at all possible relative velocities
            _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx, self.ntheta))
            _Sig[:] = (v_v*sigma_el_h_hh(v_v2*(0.5*CONST.H_MASS*(self.vth**2)/CONST.Q))).reshape(_Sig.shape, order='F')

            # Note: using H energy here for cross-section tabulated as H -> H2
            # Set SIG_H2_H = vr' x vx_vx x Integral{v_v * sigma_H2_H} over theta = 0, 2pi times differential velocity space element Vr'2pidVr'*dVx
            SIG_H2_H = np.zeros((self.nvr*self.nvx, self.nvr*self.nvx))
            SIG_H2_H[:] = (Vr2pidVrdVx*vx_vx*((_Sig @ self.dtheta).reshape(vx_vx.shape, order='F'))).reshape(SIG_H2_H.shape, order='F')

            # SIG_H2_H is now vr' * vx_vx * sigma_H2_H(v_V) ( integrated over theta ) for all possible ([vr, vx], [vr', vx'])

        if Do_SIG_H2_P == 1:
            if debrief > 1:
                print(self.prompt, 'Computing SIG_H2_P') 
            #   Compute SIG_H2_P for present velocity space grid, if it is needed and has not 
            # already been computed with the present input parameters

            # Compute sigma_H2_P * v_v at all possible relative velocities
            _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx, self.ntheta))
            _Sig[:] = (v_v*sigma_el_p_hh(v_v2*(0.5*CONST.H_MASS*(self.vth**2)/CONST.Q))).reshape(_Sig.shape, order='F')
            #energy = v_v2*(0.5*CONST.H_MASS*Vth2/CONST.Q) #NOTE is this completely unused?

            # Note: using H energy here for cross-section tabulated as p -> H2

            # Set SIG_H2_P = vr' x vx_vx x Integral{v_v * sigma_H2_P} over theta = 0, 2pi times differential velocity space element Vr'2pidVr' * dVx
            SIG_H2_P = np.zeros((self.nvr*self.nvx, self.nvr*self.nvx))
            SIG_H2_P[:] = (Vr2pidVrdVx*vx_vx*(_Sig @ self.dtheta).reshape(vx_vx.shape, order='F')).reshape(SIG_H2_P.shape, order='F')

            # SIG_H2_P is now vr' * vx_vx * sigma_h2_P(v_v) * v_v (integrated over theta) for all possible ([vr, vx], [vr', vx'])

        if Do_SIG_H2_H2 == 1:
            if debrief > 1:
                print(self.prompt, 'Computing SIG_H2_H2')
            
            #   Compute SIG_H2_H2 for present velocity space grid, if it is needed and has not 
            # already been computed with the present input parameters

            # Compute sigma_H2_H2 * vr2_vx2 * v_v at all possible relative velocities 
            _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx, self.ntheta))
            _Sig[:] = (vr2_vx2*v_v*sigma_el_hh_hh(v_v2*(CONST.H_MASS*self.mu*(self.vth**2)/CONST.Q), vis = 1)/8.0).reshape(_Sig.shape, order='F')

            # Note : For viscosity, the cross section for D -> D is the same function of 
            # center of mass energy as H -> H.

            # Set SIG_H2_H2 = vr' x Integral{vr2_vx2*v_v*sigma_H2_H2} over theta=0,2pi times differential velocity space element Vr'2pidVr'*dVx'
            SIG_H2_H2 = np.zeros((self.nvr * self.nvx, self.nvr * self.nvx))
            SIG_H2_H2[:] = (Vr2pidVrdVx*((_Sig @ self.dtheta).reshape(Vr2pidVrdVx.shape, order='F'))).reshape(SIG_H2_H2.shape, order='F')

            # SIG_H2_H2 is now vr' * sigma_H2_H2(v_v) * vr2_vx2 * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])
        
        if Do_Alpha_H2_H == 1: #NOTE Not Tested Yet
            if debrief > 1:
                print(self.prompt, 'Computing Alpha_H2_H')
            
            # Compute Alpha_H2_H for inputed fH, if it is needed and has not
            # already been computed with the present input parameters

            Alpha_H2_H = np.zeros((self.nvr, self.nvx, self.nx))
            for k in range(0, self.nx):
                Work = fH[:,:,k].reshape((self.nvr*self.nvx), order='F')
                Alpha_H2_H[:,:,k] = (SIG_H2_H @ Work).reshape(Alpha_H2_H[:,:,k].shape, order='F')

            
        # Compute nH2
        for k in range(0, self.nx):
            self.H_Moments.nH[k] = np.sum(self.dvr_vol*(fH2[:,:,k] @ self.dvx))

        if New_H2_Seed:
            MH2_H2_sum = np.zeros((self.nvr,self.nvx,self.nx))
            Delta_nH2s = 1.0

        gamma_wall = np.zeros((self.nvr,self.nvx,self.nx))
        for k in range(0, self.nx):
            if self.mesh.PipeDia[k] > 0.0:
                for j in range(0, self.nvx):
                    gamma_wall[:,j,k] = 2*self.mesh.vr/self.mesh.PipeDia[k]

        # Set iteration Scheme 
        fH2_iterate = 0 
        if (self.COLLISIONS.H2_H2_EL != 0) or (self.COLLISIONS.H2_P_CX != 0) or (self.COLLISIONS.H2_H_EL != 0) or (self.COLLISIONS.H2_P_EL != 0) or ni_correct:
            fH2_iterate=1
        fH2_generations = 0
        if fH2_iterate != 0:
            fH2_generations = 1

        # This is the iteration entry point for fH2, THP and nHP iteration.
        # Save 'seed' values for comparison later
        fH2G = np.zeros((self.nvr,self.nvx,self.nx))
        NH2G = np.zeros((self.nx, max_gen+1))
        do_fH2_iterate = True
        while do_fH2_iterate:
            do_fH2_iterate = False
            nH2s = copy.copy(self.H_Moments.nH)

            # Compute Alpha_CX for present THP and nHP, if it is needed and has not
            # already been computed with the present parameters
            if Do_Alpha_CX == 1: #NOTE Should this be in the loop? Outside loop in KH
                if debrief > 1:
                    print(self.prompt, 'Computing Alpha_CX')
                # Set Maxwellian Molecular Ion Distrobution Function (assumed to be drifting with ion velocity, vxi)
                vx_shift = self.vxi
                Tmaxwell = THP
                mol = 2
                Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
                fHp_hat = copy.copy(Maxwell)

                if self.COLLISIONS.Simple_CX:
                    # Option (B) : Use Maxwellian weighted <sigma v>
                    
                    # THP/mu at each mesh point
                    THP_mu = np.zeros((self.nvr, self.nvx, self.nx))
                    for k in range(0, self.nx):
                        THP_mu[:,:,k] = THP[k]/self.mu

                    # Molecular Charge Exchange sink rate 
                    Alpha_CX = sigmav_cx_hh(THP_mu, EH2_P)/self.vth
                    for k in range(0, self.nx):
                        Alpha_CX[:,:,k] = Alpha_CX[:,:,k]*nHP[k]
                else:
                    # Option (A): Compute SigmaV_CX from sigma directly via SIG_CX
                    Alpha_CX = np.zeros((self.nvr, self.nvx, self.nx))
                    for k in range(0, self.nx):
                        Work = (fHp_hat[:,:,k]*nHP[k]).reshape((self.nvr*self.nvx), order='F')
                        Alpha_CX[:,:,k] = SIG_CX @ Work
                    if self.Do_Alpha_CX_Test:
                        alpha_cx_test = sigmav_cx_hh(THP_mu, EH2_P)/self.vth
                        for k in range(0, self.nx):
                            alpha_cx_test[:,:,k] = alpha_cx_test[:,:,k]*nHP[k]
                            print('Compare alpha_cx and alpha_cx_test')
                            input()

            # Compute Alpha_H2_P for present Ti and ni (optionally correcting for nHP), 
            # if it is needed and has not already been computed with the present parameter
            if Do_Alpha_H2_P == 1: #NOTE Should this be in the loop? Outside loop in KH
                if debrief > 1:
                    print(self.prompt, 'Computing Alpha_H2_P')
                Alpha_H2_P = np.zeros((self.nvr, self.nvx, self.nx))
                ni = self.mesh.ne
                if ni_correct:
                    ni = np.maximum((self.mesh.ne-nHP), 0)
                # print("ni", ni)
                for k in range(0, self.nx):
                    Work = (fi_hat[:,:,k]*ni[k]).reshape((self.nvr*self.nvx), order='F')
                    Alpha_H2_P[:,:,k] = (SIG_H2_P @ Work).reshape(Alpha_H2_P[:,:,k].shape, order='F')
            
            # Compute Omega values if nH2 is non-zero 

            Omega_H2_P = np.zeros(self.nx)
            Omega_H2_H = np.zeros(self.nx)
            Omega_H2_H2 = np.zeros(self.nx)
            ii = np.argwhere(self.H_Moments.nH <= 0) 
            if ii.size <= 0: #NOTE NOT Tested yet, return on iteration
                # compute VxH2
                if self.COLLISIONS.H2_P_EL or self.COLLISIONS.H2_H_EL or self.COLLISIONS.H2_H2_EL:
                    for k in range(0, self.nx):
                        self.H_Moments.VxH[k] = self.vth*np.sum(self.dvr_vol*(fH2[:,:,k] @ (self.mesh.vx*self.dvx))) / self.H_Moments.nH[k]

                # compute Omega_H2_P for present fH2 and Alpha_H2_P if H2_P elastic collisions are included
                if self.COLLISIONS.H2_P_EL:
                    if debrief > 1:
                        print(self.prompt, 'Computing Omega_H2_P')
                    for k in range(0, self.nx):
                        DeltaVx = (self.H_Moments.VxH[k] - self.vxi[k])/self.vth
                        MagDeltaVx = np.maximum(np.abs(DeltaVx), self.DeltaVx_tol)
                        DeltaVx = np.sign(DeltaVx)*MagDeltaVx
                        Omega_H2_P[k] = np.sum(self.dvr_vol*(((Alpha_H2_P[:,:,k]*fH2[:,:,k]) @ self.dvx)))/(self.H_Moments.nH[k]*DeltaVx)
                    Omega_H2_P =  np.maximum(Omega_H2_P, 0)

                # Compute Omega_H2_H for present fH2 and Alpha_H2_H if H2_H elastic collisions are included
                if self.COLLISIONS.H2_H_EL: #NOTE Note Tested Yet
                    if debrief>1:
                        print(self.prompt+'Computing Omega_H2_H')
                    for k in range(self.nx):
                        DeltaVx = (self.H_Moments.VxH[k] - VxH[k])/self.vth
                        MagDeltaVx = np.maximum(np.abs(DeltaVx), self.DeltaVx_tol)
                        DeltaVx = np.sign(DeltaVx)*MagDeltaVx
                        Omega_H2_H[k] = np.sum(self.dvr_vol*((Alpha_H2_H[:,:,k]*fH2[:,:,k]) @ self.dvx)/(self.H_Moments.nH[k]*DeltaVx))
                    Omega_H2_H = np.maximum(Omega_H2_H, 0)

                # Compute Omega_H2_H2 for present fH2 if H2_H2 elastic collisions are included
                if self.COLLISIONS.H2_H2_EL:
                    if debrief > 1:
                        print(self.prompt, 'Computing Omega_H2_H2')

                    Wperp_paraH2 = np.zeros(self.nx)
                    vr2_2vx_ran2 = np.zeros((self.nvr,self.nvx))
                    if np.sum(MH2_H2_sum) < 0:
                        for k in range(0, self.nx):
                            for i in range(0, self.nvr):
                                vr2_2vx_ran2[i,:] = self.mesh.vr[i]**2 - 2*(self.mesh.vx - self.H_Moments.VxH[k]/self.vth)**2
                            Wperp_paraH2[k] = np.sum(self.dvr_vol*((vr2_2vx_ran2*fH2[:,:,k]) @ self.dvx))/self.H_Moments.nH[k]
                    else:
                        for k in range(0, self.nx):
                            M_fH2 = MH2_H2_sum[:,:,k] - fH2[:,:,k]
                            Wperp_paraH2[k] = -np.sum(self.dvr_vol*((self.vr2_2vx2_2D*M_fH2) @ self.dvx))/self.H_Moments.nH[k]

                    for k in range(0, self.nx):
                        Work = fH2[:,:,k].reshape((self.nvr*self.nvx), order='F')
                        Alpha_H2_H2 = (SIG_H2_H2 @ Work).reshape((self.nvr,self.nvx), order='F')
                        Wpp = Wperp_paraH2[k]
                        MagWpp = np.maximum(abs(Wpp), self.Wpp_tol)
                        Wpp = np.sign(Wpp)*MagWpp  
                        Omega_H2_H2[k] = np.sum(self.dvr_vol*((Alpha_H2_H2*Work.reshape((self.nvr,self.nvx), order='F')) @ self.dvx))/(self.H_Moments.nH[k]*Wpp)
                        
                    Omega_H2_H2 = np.maximum(Omega_H2_H2, 0)

            # Total Elastic scattering frequency
            Omega_EL = Omega_H2_P + Omega_H2_H + Omega_H2_H2

            # Total collision frequency
            alpha_c = np.zeros((self.nvr,self.nvx,self.nx))
            if self.COLLISIONS.H2_P_CX:
                for k in range(0, self.nx):
                    alpha_c[:,:,k] = Alpha_CX[:,:,k]+Alpha_Loss[k]+Omega_EL[k]+gamma_wall[:,:,k]
            else: 
                for k in range(0, self.nx):
                    alpha_c[:,:,k] = Alpha_Loss[k]+Omega_EL[k]+gamma_wall[:,:,k]

            # Test x grid spacing based on Eq.(27) in notes
            if debrief > 1: 
                print(self.prompt, 'Testing x grid spacing')
            self.Errors.Max_dx = np.full(self.nx, 1.0E32)
            for k in range(0, self.nx) : 
                for j in self.vx_pos:
                    denom = alpha_c[:,j,k]
                    self.Errors.Max_dx[k] = np.minimum(self.Errors.Max_dx[k], np.min(2*self.mesh.vx[j]/denom))

            dx = np.roll(self.mesh.x,-1) - self.mesh.x
            Max_dxL = self.Errors.Max_dx[0:self.nx-1]
            Max_dxR = self.Errors.Max_dx[1:self.nx]
            self.Errors.Max_dx = np.minimum(Max_dxL, Max_dxR)
            ilarge = np.argwhere(self.Errors.Max_dx < dx[0:self.nx-1])

            if ilarge.size > 0:
                print(self.prompt,'x mesh spacing is too large!') #NOTE Check Formatting
                debug = 1
                out = ''
                jj = 0
                print(' x(k+1)-x(k)  Max_dx(k)   x(k+1)-x(k)  Max_dx(k)   x(k+1)-x(k)  Max_dx(k)   x(k+1)-x(k)  Max_dx(k)   x(k+1)-x(k)  Max_dx(k)')
                for ii in range(0, np.size(ilarge)-1):
                    jj = jj + 1
                    out = out + str(ilarge[ii]) +' '+ str(self.mesh.x(ilarge[ii]+1)-self.mesh.x(ilarge[ii])) +' '+ self.Errors.Max_dx(ilarge[ii]) +' ' # I didn't include any of the formatting from the original code I thought this is something we could determine later
                    if jj > 4:
                        print(out)
                        jj = 0 
                        out = ''
                if jj > 0: 
                    print(out)
                raise Exception("x mesh spacing is too large") 
            
            # Define parameters Ak, Bk, Ck, Dk, Fk, Gk
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
                    Fk[:,j,k] = (self.mesh.x[k+1] - self.mesh.x[k])*fw_hat[:,j]*(SH2[k+1]+SH2[k]) / (self.vth*denom)
            for k in range(1, self.nx):
                for j in self.vx_neg:
                    denom = -2*self.mesh.vx[j] + (self.mesh.x[k] - self.mesh.x[k-1])*alpha_c[:,j,k-1]
                    Ck[:,j,k] = (-2*self.mesh.vx[j] - (self.mesh.x[k] - self.mesh.x[k -1])*alpha_c[:,j,k]) / denom
                    Dk[:,j,k] = (self.mesh.x[k] - self.mesh.x[k-1]) / denom
                    Gk[:,j,k] = (self.mesh.x[k] - self.mesh.x[k-1])*fw_hat[:,j]*(SH2[k]+SH2[k-1]) / (self.vth*denom)

            # Compute first-flight (0th generation) neutral distribution function
            Swall_sum = np.zeros((self.nvr,self.nvx,self.nx))
            Beta_CX_sum = np.zeros((self.nvr,self.nvx,self.nx))
            MH2_P_sum = np.zeros((self.nvr,self.nvx,self.nx))
            MH2_H_sum = np.zeros((self.nvr,self.nvx,self.nx))
            MH2_H2_sum = np.zeros((self.nvr,self.nvx,self.nx))
            igen = 0
            if debrief > 0:
                print(self.prompt, 'Computing molecular neutral generation#', sval(igen))
            fH2G[:,self.vx_pos,0] = fH2[:,self.vx_pos,0]
            for k in range(0, self.nx-1):
                fH2G[:,self.vx_pos,k+1] = fH2G[:,self.vx_pos,k]*Ak[:,self.vx_pos,k] + Fk[:,self.vx_pos,k]
            for k in range(self.nx-1,0,-1):
                fH2G[:,self.vx_neg,k-1] = fH2G[:,self.vx_neg,k]*Ck[:,self.vx_neg,k] + Gk[:,self.vx_neg,k]

            # Compute first-flight neutral density profile 
            for k in range(0, self.nx):
                NH2G[k, igen] = np.sum(self.dvr_vol*(fH2G[:,:,k] @ self.dvx))

            # NOTE Implement Plotting Later

            # Set total molecular neutral distrobution function to first flight generation 
            fH2 = copy.copy(fH2G)
            self.H_Moments.nH = NH2G[:,0]
            
    # next_generation #########################################################################################################################################################################
            while True:
                if igen+1 > max_gen or fH2_generations == 0: 
                    if debrief > 1:
                        print(self.prompt,'Completed ', sval(max_gen), ' generations. Returning present solution...')
                    break
                igen = igen + 1
                if debrief > 0: 
                    print(self.prompt, 'Computing molecular neutral generation#', sval(igen))
            
                #Compute Swall from previous generation
                Swall = np.zeros((self.nvr, self.nvx, self.nx))
                if np.sum(gamma_wall) > 0:
                    if debrief > 1:
                        print(self.prompt, 'Computing Swall')
                    for k in range(0, self.nx): 
                        Swall[:,:k] = fw_hat*np.sum(self.dvr_vol*((gamma_wall[:,:,k]*fH2G[:,:,k]) @ self.dvx))
                    #Sum wall collision source over all generations
                    Swall_sum = Swall_sum + Swall

                #Compute Beta_CX from previous generation
                Beta_CX = np.zeros((self.nvr,self.nvx,self.nx))
                if self.COLLISIONS.H2_P_CX: 
                    if debrief > 1:
                        print(self.prompt, 'Computing Beta_CX')
                    if self.COLLISIONS.Simple_CX:
                        # Option (B): Compute charge exchange source with assumption that CX source neutrals have 
                        # molecular ion distribution function
                        for k in range(0, self.nx): 
                            Beta_CX[:,:,k] = fHp_hat[:,:,k]*np.sum(self.dvr_vol*((Alpha_CX[:,:,k]*fH2G[:,:,k]) @ self.dvx))
                    else: 
                        # Option (A): Compute charge exchange source using fH2 and vr x sigma x v_v at each velocity mesh point
                        for k in range(0, self.nx):
                            Work = fH2G[:,:,k]
                            Beta_CX[:,:,k] = nHP[k]*fHp_hat[:,:,k]*(SIG_CX @ Work)
                    #Sum 
                    Beta_CX_sum = Beta_CX_sum + Beta_CX

                # Compute MH2 from previous generation
                MH2_H2 = np.zeros((self.nvr,self.nvx,self.nx))
                MH2_P = np.zeros((self.nvr,self.nvx,self.nx))
                MH2_H = np.zeros((self.nvr,self.nvx,self.nx))
                OmegaM = np.zeros((self.nvr,self.nvx,self.nx))
                if self.COLLISIONS.H2_H2_EL or self.COLLISIONS.H2_P_EL or self.COLLISIONS.H2_H_EL:
                    # Compute VxH2G, TH2G

                    VxH2G = np.zeros(self.nx)
                    TH2G = np.zeros(self.nx)
                    vr2vx2_ran2 = np.zeros((self.nvr,self.nvx))
                    for k in range(0, self.nx):
                        VxH2G[k] = self.vth*np.sum(self.dvr_vol*(fH2G[:,:,k] @ (self.mesh.vx * self.dvx))) / NH2G[k,igen-1]
                        for i in range(0, self.nvr):
                            vr2vx2_ran2[i,:] = self.mesh.vr[i]**2 + (self.mesh.vx - VxH2G[k]/self.vth)**2
                        TH2G[k] = (2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((vr2vx2_ran2*fH2G[:,:,k]) @ self.dvx)) / (3*CONST.Q*NH2G[k,igen-1])

                    if self.COLLISIONS.H2_H2_EL:
                        if debrief > 1: 
                            print(self.prompt, 'Computing MH2_H2')
                        # Compute MH2_H2
                        vx_shift = VxH2G
                        Tmaxwell = TH2G
                        mol = 2
                        Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)

                        for k in range(0, self.nx):
                            MH2_H2[:,:,k] = Maxwell[:,:,k]*NH2G[k,igen-1]
                            OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H2_H2[k]*MH2_H2[:,:,k]
                        MH2_H2_sum = MH2_H2_sum + MH2_H2

                    if self.COLLISIONS.H2_P_EL:
                        if debrief > 1:
                            print(self.prompt, 'Computing MH2_P')
                        # Compute MH2_P
                        vx_shift = (2 * VxH2G + self.vxi)/3
                        Tmaxwell = TH2G + (4/9)*(self.mesh.Ti - TH2G + ((self.mu*CONST.H_MASS*(self.vxi - VxH2G)**2)/(6*CONST.Q)))
                        mol = 2
                        Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)

                        for k in range(0, self.nx):
                            MH2_P[:,:,k] = Maxwell[:,:,k]*NH2G[k,igen-1]
                            OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H2_P[k]*MH2_P[:,:,k]
                        MH2_P_sum = MH2_P_sum + MH2_P

                    if self.COLLISIONS.H2_H_EL: #NOTE Not Tested Yet
                        if debrief > 1:
                            print(self.prompt, 'Computing MH2_H')
                        #Compute MH2_H
                        vx_shift = (2*VxH2G + VxH)/3
                        Tmaxwell = TH2G + (4/9)*(TH - TH2G + ((self.mu*CONST.H_MASS*(VxH - VxH2G)**2)/(6*CONST.Q)))
                        mol = 2
                        Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)

                        for k in range(0, self.nx):
                            MH2_H[:,:,k] = Maxwell[:,:,k]*NH2G[k,igen-1]
                            OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H2_H[k]*MH2_H[:,:,k]
                        MH2_H_sum = MH2_H_sum + MH2_H

                # Compute next generation molecular distribution
                fH2G[:] = 0.0
                for k in range(0, self.nx-1):
                    fH2G[:,self.vx_pos,k+1] = Ak[:,self.vx_pos,k]*fH2G[:,self.vx_pos,k] + Bk[:,self.vx_pos,k]*(Swall[:,self.vx_pos,k+1] + Beta_CX[:,self.vx_pos,k+1] + OmegaM[:,self.vx_pos,k+1] + Swall[:,self.vx_pos,k] + Beta_CX[:,self.vx_pos,k] + OmegaM[:,self.vx_pos,k])
                for k in range(self.nx-1, 0, -1):
                    fH2G[:,self.vx_neg,k-1] = Ck[:,self.vx_neg,k]*fH2G[:,self.vx_neg,k] + Dk[:,self.vx_neg,k]*(Swall[:,self.vx_neg,k-1] + Beta_CX[:,self.vx_neg,k-1] + OmegaM[:,self.vx_neg,k-1] + Swall[:,self.vx_neg,k] + Beta_CX[:,self.vx_neg,k] + OmegaM[:,self.vx_neg,k])
                for k in range(0, self.nx):
                    NH2G[k,igen] = np.sum(self.dvr_vol*(fH2G[:,:,k] @ self.dvx))
                

                # NOTE Implement Plotting later

                # Add result to total neutral distribution function
                fH2 = fH2 + fH2G
                self.H_Moments.nH = self.H_Moments.nH + NH2G[:,igen]

                # Compute 'generation error': Delta_nH2G=max(NH2G(*,igen)/max(nH2))
                # and decide if another generation should be computed
                Delta_nH2G = np.max(NH2G[:,igen]/np.max(self.H_Moments.nH))
                if (Delta_nH2G < truncate) or (fH2_iterate and (Delta_nH2G < 0.003*Delta_nH2s)):
                    # If fH2 'seed' is being iterated, then do another generation until the 'generation error'
                    # is less than 0.003 times the 'seed error' or is less than TRUNCATE
                    break
                # If fH2 'seed' is NOT being iterated, then do another generation unitl the 'generation error'
                # is less than parameter TRUNCATE


    # fH2_done #########################################################################################################################################################################

            # NOTE Implement Plotting Later

            # Compute H2 density profile
            for k in range(0, self.nx): #NOTE Should the rest of this be in the loop? KH, nH2 is only thing computed in the loop
                self.H_Moments.nH[k] = np.sum(self.dvr_vol*(fH2[:,:,k] @ self.dvx))

            # GammaxH2 - particle flux in x direction
            GammaxH2 = np.zeros(self.nx)
            for k in range(0, self.nx):
                GammaxH2[k] = self.vth*np.sum(self.dvr_vol*(fH2[:,:,k] @ (self.mesh.vx*self.dvx)))

            # VxH2 - x velocity
            self.H_Moments.VxH = GammaxH2 / self.H_Moments.nH
            _VxH2 = self.H_Moments.VxH / self.vth 

            # magnitude of random velocity at each mesh point 
            vr2vx2_ran = np.zeros((self.nvr, self.nvx, self.nx))
            for i in range(0, self.nvr):
                for k in range(0, self.nx):
                    vr2vx2_ran[i,:,k] = self.mesh.vr[i]**2 + (self.mesh.vx - _VxH2[k])**2

            # pH2 - pressure 
            pH2 = np.zeros(self.nx)
            for k in range(0, self.nx):
                pH2[k] = ((2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((vr2vx2_ran[:,:,k]*fH2[:,:,k]) @ self.dvx)))/(3*CONST.Q)

            #TH2 - temperature 
            self.H_Moments.TH = pH2/self.H_Moments.nH

            # Compute NuDis - Dissociation frequency 
            NuDis = self.mesh.ne*np.sum(sigv[:,7:11], 1)
            
            # Compute NuE (assume np=ne) - Energy equilibration frequency H(+) <-> H2(+)
            NuE = (7.7e-7*self.mesh.ne*1.0e-6)/(np.sqrt(self.mu)*(self.mesh.Ti**1.5))
            
            # Compute H2(+) density profile
            nHP = (self.H_Moments.nH*self.mesh.ne*sigv[:,1])/(NuDis + self.NuLoss)

            # Compute THP - temperature of molecular ions
            THP = (self.mesh.Ti*NuE)/(NuE + NuDis + self.NuLoss)
            
            if fH2_iterate:
                # Compute 'seed error': Delta_nH2s=(|nH2s-nH2|)/max(nH2) 
                # If Delta_nH2s is greater than 10*truncate then iterate fH2

                Delta_nH2s = np.max(np.abs(nH2s - self.H_Moments.nH))/np.max(self.H_Moments.nH)
                if Delta_nH2s > 10*truncate:
                    do_fH2_iterate = True
        
        # Update Swall_sum using last generation
        Swall = np.zeros((self.nvr, self.nvx, self.nx))
        if np.sum(gamma_wall) > 0:
            for k in range(0, self.nx):
                Swall[:,:,k] = fw_hat * np.sum(self.dvr_vol*((gamma_wall[:,:,k]*fH2G[:,:,k]) @ self.dvx))
                Swall_sum = Swall_sum + Swall
        
        # Update Beta_CX_sum using last generation
        Beta_CX = np.zeros((self.nvr, self.nvx, self.nx))
        if self.COLLISIONS.H2_P_CX:
            if debrief > 1:
                print(self.prompt, 'Computing Beta_CX')
            if self.COLLISIONS.Simple_CX:
                # Option (B): Compute charge exchange source with assumption that CX source neutrals have
                # molecular ion distribution function
                for k in range(0, self.nx):
                    Beta_CX[:,:,k] = fHp_hat[:,:,k]*np.sum(self.dvr_vol*(Alpha_CX[:,:,k]*fH2G[:,:,k] @ self.dvx))
            else: #NOTE Not Tested
                # Option (A): Compute charge exchange source using fH2 and vr x sigma x v_v at each velocity mesh point
                for k in range(0, self.nx):
                    Work = fH2G[:,:,k]
                    Beta_CX[:,:,k] = nHP[k]*fHp_hat[:,:,k]*(SIG_CX @ Work)
            Beta_CX_sum = Beta_CX_sum + Beta_CX

        # Update MH2_*_sum using last generation
        MH2_H2 = np.zeros((self.nvr,self.nvx,self.nx))
        MH2_P = np.zeros((self.nvr,self.nvx,self.nx))
        MH2_H = np.zeros((self.nvr,self.nvx,self.nx))
        OmegaM = np.zeros((self.nvr,self.nvx,self.nx))
        if self.COLLISIONS.H2_H2_EL or self.COLLISIONS.H2_P_EL or self.COLLISIONS.H2_H_EL: 
            # Compute VxH2G, TH2G
            VxH2G = np.zeros(self.nx)
            TH2G = np.zeros(self.nx)
            vr2vx2_ran2 = np.zeros((self.nvr,self.nvx))
            for k in range(0, self.nx):
                VxH2G[k] = self.vth*np.sum(self.dvr_vol*(fH2G[:,:,k] @ (self.mesh.vx*self.dvx))) / NH2G[k,igen]
                for i in range(0, self.nvr):
                    vr2vx2_ran2[i,:] = self.mesh.vr[i]**2 + (self.mesh.vx - VxH2G[k]/self.vth)**2
                TH2G[k] = (2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((vr2vx2_ran2*fH2G[:,:,k]) @ self.dvx))/(3*CONST.Q*NH2G[k,igen])

            if self.COLLISIONS.H2_H2_EL:
                if debrief > 1: 
                    print(self.prompt, 'Computing MH2_H2')
                # Compute MH2_H2
                vx_shift = VxH2G
                Tmaxwell = np.copy(TH2G)
                mol = 2
                Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
                for k in range(0, self.nx):
                    MH2_H2[:,:,k] = Maxwell[:,:,k]*NH2G[k,igen]
                    OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H2_H2[k]*MH2_H2[:,:,k]
                MH2_H2_sum = MH2_H2_sum + MH2_H2

            if self.COLLISIONS.H2_P_EL:
                if debrief > 1:
                    print(self.prompt, 'Computing MH2_P')
                # Compute MH2_P
                vx_shift = (2*VxH2G + self.vxi)/3
                Tmaxwell = TH2G + (4/9)*(self.mesh.Ti - TH2G + self.mu*CONST.H_MASS*((self.vxi - VxH2G)**2) / (6*CONST.Q))
                mol = 2
                Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
                
                for k in range(0, self.nx):
                    MH2_P[:,:,k] = Maxwell[:,:,k]*NH2G[k,igen]
                    OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H2_P[k]*MH2_P[:,:,k]
                MH2_P_sum = MH2_P_sum + MH2_P

                if self.COLLISIONS.H2_H_EL: #NOTE Not Tested Yet
                    if debrief > 1:
                        print(self.prompt, 'Computing MH2_H')
                    # Compute MH2_H
                    vx_shift = (2*VxH2G + VxH)/3
                    Tmaxwell = TH2G + (4/9)*(TH - TH2G + self.mu*CONST.H_MASS*((VxH - VxH2G)**2)/(6*CONST.Q))
                    mol = 2
                    Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
                    
                    for k in range(0, self.nx):
                        MH2_H[:,:,k] = Maxwell[:,:,k]*NH2G[k,igen]
                        OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H2_H[k]*MH2_H[:,:,k]
                    MH2_H_sum = MH2_H_sum + MH2_H
            
        # Compute remaining moments
        #NOTE Why are these seperate loops?
        # piH2_xx
        for k in range(0, self.nx):
            self.Output.piH2_xx[k] = (((2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*(fH2[:,:,k] @ (self.dvx*(self.mesh.vx - _VxH2[k])**2))))/CONST.Q) - pH2[k]
        # piH2_yy
        for k in range(0, self.nx):
            self.Output.piH2_yy[k] = (((2*self.mu*CONST.H_MASS)*(self.vth**2)*0.5*np.sum((self.dvr_vol*(self.mesh.vr**2))*(fH2[:,:,k] @ self.dvx)))/CONST.Q) - pH2[k]
        # piH2_zz 
        self.Output.piH2_zz = copy.copy(self.Output.piH2_yy) 
        # qxH2
        qxH2 = np.zeros(self.nx)
        for k in range(0, self.nx):
            qxH2[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**3)*np.sum(self.dvr_vol*((vr2vx2_ran[:,:,k]*fH2[:,:,k]) @ (self.dvx*(self.mesh.vx - _VxH2[k]))))
        
        # C = RHS of Boltzman equation for total fH2
        Sloss = np.zeros(self.nx)
        WallH2 = np.zeros(self.nx)
        QH2 = np.zeros(self.nx)
        RxH2 = np.zeros(self.nx)
        for k in range(0, self.nx):
            C = self.vth*((fw_hat[:,:]*SH2[k]/self.vth) + Swall_sum[:,:,k] + Beta_CX_sum[:,:,k] - (alpha_c[:,:,k]*fH2[:,:,k]) + \
                    (Omega_H2_P[k]*MH2_P_sum[:,:,k]) + (Omega_H2_H[k]*MH2_H_sum[:,:,k]) + (Omega_H2_H2[k]*MH2_H2_sum[:,:,k]))
            QH2[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((vr2vx2_ran[:,:,k]*C) @ self.dvx))
            RxH2[k] = (2*self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_vol*(C @ (self.dvx*(self.mesh.vx - _VxH2[k]))))
            Sloss[k] = -np.sum(self.dvr_vol*(C @ self.dvx)) + SH2[k]
            WallH2[k] = np.sum(self.dvr_vol*((gamma_wall[:,:,k]*fH2[:,:,k]) @ self.dvx))

            if self.COLLISIONS.H2_H_EL:
                CH2_H = self.vth*Omega_H2_H[k]*(MH2_H_sum[:,:,k] - fH2[:,:,k])
                self.Output.RxH_H2[k] = (2*self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_vol*(CH2_H @ (self.dvx*(self.mesh.vx - _VxH2[k]))))
                self.Output.EH_H2[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((vr2vx2[:,:,k]*CH2_H) @ self.dvx))

            if self.COLLISIONS.H2_P_EL:
                CH2_P = self.vth*Omega_H2_P[k]*(MH2_P_sum[:,:,k] - fH2[:,:,k])
                self.Output.RxP_H2[k] = (2*self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_vol*(CH2_P @ (self.dvx*(self.mesh.vx - _VxH2[k]))))
                self.Output.EP_H2[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((vr2vx2[:,:,k]*CH2_P) @ self.dvx))

            if self.COLLISIONS.H2_P_CX:
                CH2_HP_CX = self.vth*(Beta_CX_sum[:,:,k] - Alpha_CX[:,:,k]*fH2[:,:,k])
                self.Output.RxH2CX[k] = (2*self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_vol*(CH2_HP_CX @ (self.dvx*(self.mesh.vx - _VxH2[k]))))
                self.Output.EH2CX[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((vr2vx2[:,:,k]*CH2_HP_CX) @ self.dvx))

            CW_H2 = self.vth*(Swall_sum[:,:,k] - gamma_wall[:,:,k]*fH2[:,:,k])
            self.Output.RxW_H2[k] = (2*self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_vol*(CW_H2 @ (self.dvx*(self.mesh.vx - _VxH2[k]))))
            self.Output.EW_H2[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((vr2vx2[:,:,k]*CW_H2) @ self.dvx))

            if self.COLLISIONS.H2_H2_EL:
                vr2_2vx_ran2 = np.zeros((self.nvr,self.nvx))
                CH2_H2 = self.vth*Omega_H2_H2[k]*(MH2_H2_sum[:,:,k] - fH2[:,:,k])
                for i in range(0, self.nvr):
                    vr2_2vx_ran2[i,:] = self.mesh.vr[i]**2 - 2*((self.mesh.vx - _VxH2[k])**2)
                self.Output.Epara_PerpH2_H2[k] = -0.5*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((vr2_2vx_ran2*CH2_H2) @ self.dvx))

        # qxH2_total
        qxH2_total = (0.5*self.H_Moments.nH*(2*self.mu*CONST.H_MASS)*self.H_Moments.VxH*self.H_Moments.VxH + 2.5*pH2*CONST.Q)*self.H_Moments.VxH + CONST.Q*self.Output.piH2_xx*self.H_Moments.VxH + qxH2

        # QH2_total
        QH2_total = QH2 + RxH2*self.H_Moments.VxH - 0.5*(2*self.mu*CONST.H_MASS)*(Sloss - SH2)*self.H_Moments.VxH*self.H_Moments.VxH

        # Albedo
        gammax_plus = self.vth*np.sum(self.dvr_vol*(fH2[:,self.vx_pos,0] @ (self.mesh.vx[self.vx_pos]*self.dvx[self.vx_pos]))) 
        gammax_minus = self.vth*np.sum(self.dvr_vol*(fH2[:,self.vx_neg,0] @ (self.mesh.vx[self.vx_neg]*self.dvx[self.vx_neg])))
        AlbedoH2 = 0.0
        if np.abs(gammax_plus) > 0:
            AlbedoH2 = -gammax_minus/gammax_plus

        # Compute Mesh Errors
        self.Errors.mesh_error = np.zeros((self.nvr,self.nvx,self.nx))
        max_mesh_error = 0.0
        min_mesh_error = 0.0
        mtest = 5
        moment_error = np.zeros((self.nx,mtest))
        max_moment_error = np.zeros(mtest)
        self.Errors.C_Error = np.zeros(self.nx)
        self.Errors.CX_Error = np.zeros(self.nx)
        Wall_error = np.zeros(self.nx)
        self.Errors.H2_H2_error = np.zeros((self.nx, 3))
        H2_H_error = np.zeros((self.nx, 3))
        H2_P_error = np.zeros((self.nx, 3))
        max_H2_H2_error = np.zeros(3)
        max_H2_H_error = np.zeros(3)
        max_H2_P_error = np.zeros(3)

        if compute_errors: #NOTE Simplify these with lambda functions, not norm and error calculation are repeated and unneccesary

            if debrief > 1:
                print(self.prompt, 'Computing Collision Operator, Mesh, and Moment Normalized Errors')

            Sloss2 = self.vth*Alpha_Loss*self.H_Moments.nH 
            for k in range(0, self.nx):
                self.Errors.C_Error[k] = np.abs(Sloss[k] - Sloss2[k])/np.max(np.abs(np.array([Sloss[k], Sloss2[k]])))

            # Test conservation of particles for charge exchange operator
            if self.COLLISIONS.H2_P_CX:
                for k in range(0, self.nx):
                    CX_A = np.sum(self.dvr_vol*((Alpha_CX[:,:,k]*fH2[:,:,k]) @ self.dvx))
                    CX_B = np.sum(self.dvr_vol*(Beta_CX_sum[:,:,k] @ self.dvx))
                    self.Errors.CX_Error[k] = np.abs(CX_A - CX_B)/np.max(np.abs(np.array([CX_A, CX_B])))

            # Test conservation of particles for wall collision operator
            if np.sum(self.mesh.PipeDia) > 0: #NOTE Not Tested Yet
                for k in range(0, self.nx):
                    Wall_A = WallH2[k]
                    Wall_B = np.sum(self.dvr_vol*(Swall_sum[:,:,k] @ self.dvx))
                    if np.max(np.abs(np.array([Wall_A, Wall_B]))) > 0:
                        Wall_error[k] = np.abs(Wall_A - Wall_B)/np.max(np.abs(np.array([Wall_A, Wall_B])))

            # Test conservation of particles, x momentum, and total energy of elastic collision operators
            for m in range(0, 3):
                for k in range(0, self.nx):
                    if m < 2:
                        TfH2 = np.sum(self.dvr_vol*(fH2[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                    else:
                        TfH2 = np.sum(self.dvr_vol*((vr2vx2[:,:,k]*fH2[:,:,k]) @ self.dvx))

                    if self.COLLISIONS.H2_H2_EL:
                        if m < 2:
                            TH2_H2 = np.sum(self.dvr_vol*(MH2_H2_sum[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                        else:
                            TH2_H2 = np.sum(self.dvr_vol*((vr2vx2[:,:,k]*MH2_H2_sum[:,:,k]) @ self.dvx))
                        self.Errors.H2_H2_error[k,m] = np.abs(TfH2 - TH2_H2)/np.max(np.abs(np.array([TfH2, TH2_H2])))
                    
                    if self.COLLISIONS.H2_H_EL:
                        if m < 2:
                            TH2_H = np.sum(self.dvr_vol*(MH2_H_sum[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                        else:
                            TH2_H = np.sum(self.dvr_vol*((vr2vx2[:,:,k]*MH2_H_sum[:,:,k]) @ self.dvx))
                        H2_H_error[k,m] = np.abs(TfH2 - TH2_H)/np.max(np.abs(np.array([TfH2, TH2_H])))

                    if self.COLLISIONS.H2_P_EL:
                        if m < 2:
                            TH2_P = np.sum(self.dvr_vol*(MH2_P_sum[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                        else:
                            TH2_P = np.sum(self.dvr_vol*((vr2vx2[:,:,k]*MH2_P_sum[:,:,k]) @ self.dvx))
                        H2_P_error[k,m] = np.abs(TfH2 - TH2_P)/np.max(np.abs(np.array([TfH2, TH2_P])))

                max_H2_H2_error[m] = np.max(self.Errors.H2_H2_error[:,m])
                max_H2_H_error[m] = np.max(H2_H_error[:,m])
                max_H2_P_error[m] = np.max(H2_P_error[:,m])

            if self.CI_Test:
                minRx = 1.0e-6
                minEpara_perp = 1.0e-6

                # Compute Momentum transfer rate via full collision integrals for charge exchange and mixed elastic scattering
                # Then compute error between this and actual momentum transfer resulting from CX and BKG (elastic) models

                if self.COLLISIONS.H2_P_CX: # H2(+) -> H2 charge exchange momentum transfer via full collision integral
                    print(self.prompt, 'Computing H2(+) -> H2 Charge Exchange Momentum Transfer')
                    _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx,self.ntheta))
                    _Sig[:] = (v_v*sigma_cx_hh(v_v2*(CONST.H_MASS*(self.vth**2)/CONST.Q))).reshape(_Sig.shape, order='F')
                    SIG_VX_CX = np.zeros((self.nvr*self.nvx,self.nvr*self.nvx))
                    SIG_VX_CX[:] = (Vr2pidVrdVx*vx_vx*((_Sig @ self.dtheta).reshape(vx_vx.shape, order='F'))).reshape(SIG_VX_CX.shape, order='F')
                    alpha_vx_cx = np.zeros((self.nvr,self.nvx,self.nx))

                    for k in range(0, self.nx):
                        Work = (nHP[k]*fHp_hat[:,:,k]).reshape((self.nvr*self.nvx), order='F')
                        alpha_vx_cx[:,:,k] = (SIG_VX_CX @ Work).reshape(alpha_vx_cx[:,:,k].shape, order='F')

                    RxCI_CX = np.zeros(self.nx)
                    for k in range(0, self.nx):
                        RxCI_CX[k] = -(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((alpha_vx_cx[:,:,k]*fH2[:,:,k]) @ self.dvx))

                    norm = np.max(np.abs(np.array([self.Output.RxH2CX, RxCI_CX])))
                    CI_CX_error = np.zeros(self.nx)
                    for k in range(0, self.nx):
                        CI_CX_error[k] = np.abs(self.Output.RxH2CX[k] - RxCI_CX[k])/norm

                    print(self.prompt,'Maximum normalized momentum transfer error in CX collision operator: ', sval(np.max(CI_CX_error)))

                if self.COLLISIONS.H2_P_EL: # P -> H2 momentum transfer via full collision integral
                    RxCI_P_H2 = np.zeros(self.nx)
                    for k in range(0, self.nx):
                        RxCI_P_H2[k] = -(1/3)*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((Alpha_H2_P[:,:,k] * fH2[:,:,k]) @ self.dvx))

                    norm = np.max(np.abs(np.array([self.Output.RxP_H2, RxCI_P_H2])))
                    CI_P_H2_error = np.zeros(self.nx)
                    for k in range(0, self.nx):
                        CI_P_H2_error[k] = np.abs(self.Output.RxP_H2[k] - RxCI_P_H2[k])/norm 

                    print(self.prompt, 'Maximum normalized momentum transfer error in P -> H2 elastic BKG collision operator: ', sval(np.max(CI_P_H2_error)))
                
                if self.COLLISIONS.H2_H_EL: # H -> H2 momentum transfer via full collision integral
                    RxCI_H_H2 = np.zeros(self.nx)
                    for k in range(0, self.nx):
                        RxCI_H_H2[k] = -(1/3)*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((Alpha_H2_H[:,:,k]*fH2[:,:,k]) @ self.dvx))
                    
                    norm = np.max(np.abs(np.array([self.Output.RxH_H2, RxCI_H_H2])))
                    CI_H_H2_error = np.zeros(self.nx)
                    for k in range(0, self.nx):
                        CI_H_H2_error[k] = np.abs(self.Output.RxH_H2[k] - RxCI_H_H2[k])/norm
                    
                    print(self.prompt, 'Maximum normalized momentum transfer error in H -> H2 elastic BKG collision operator: ', sval(np.max(CI_H_H2_error)))
                
                if self.COLLISIONS.H2_H2_EL: # H2 -> H2 perp/parallel energy transfer via full collision integral
                    Epara_Perp_CI = np.zeros(self.nx)
                    for k in range(0, self.nx):
                        Work = fH2[:,:,k].reshape((self.nvr*self.nvx), order='F')
                        Alpha_H2_H2 = (SIG_H2_H2 @ Work).reshape((self.nvr,self.nvx), order='F')
                        Epara_Perp_CI[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**3)*np.sum(self.dvr_vol*((Alpha_H2_H2*fH2[:,:,k]) @ self.dvx)) 
                    
                    norm = np.max(np.abs(np.array([self.Output.Epara_PerpH2_H2, Epara_Perp_CI])))
                    CI_H2_H2_error = np.zeros(self.nx)
                    for k in range(0, self.nx):
                        CI_H2_H2_error[k] = np.abs(self.Output.Epara_PerpH2_H2[k] - Epara_Perp_CI[k])/norm 
                    
                    print(self.prompt, 'Maximum normalized perp/parallel energy transfer error in H2 -> H2 elastic BKG collision operator: ', sval(np.max(CI_H2_H2_error)))
            
            # Mesh Point Error based on fH2 satisfying Boltzmann equation
            T1 = np.zeros((self.nvr,self.nvx,self.nx))
            T2 = np.zeros((self.nvr,self.nvx,self.nx))
            T3 = np.zeros((self.nvr,self.nvx,self.nx))
            T4 = np.zeros((self.nvr,self.nvx,self.nx))
            T5 = np.zeros((self.nvr,self.nvx,self.nx))
            T6 = np.zeros((self.nvr,self.nvx,self.nx))
            for k in range(0, self.nx-1):
                for j in range(0, self.nvx):
                    T1[:,j,k] = 2*self.mesh.vx[j]*(fH2[:,j,k+1] - fH2[:,j,k])/(self.mesh.x[k+1] - self.mesh.x[k]) 
                T2[:,:,k] = fw_hat[:,:]*(SH2[k+1] + SH2[k])/self.vth
                T3[:,:,k] = Beta_CX_sum[:,:,k+1] + Beta_CX_sum[:,:,k]
                T4[:,:,k] = alpha_c[:,:,k+1]*fH2[:,:,k+1] + alpha_c[:,:,k]*fH2[:,:,k]
                T5[:,:,k] = Omega_H2_P[k+1]*MH2_P_sum[:,:,k+1] + Omega_H2_H[k+1]*MH2_H_sum[:,:,k+1] + Omega_H2_H2[k+1]*MH2_H2_sum[:,:,k+1] + \
                        Omega_H2_P[k]*MH2_P_sum[:,:,k] + Omega_H2_H[k]*MH2_H_sum[:,:,k] + Omega_H2_H2[k]*MH2_H2_sum[:,:,k]
                T6[:,:,k] = Swall_sum[:,:,k+1] + Swall_sum[:,:,k]
                self.Errors.mesh_error[:,:,k] = np.abs(T1[:,:,k] - T2[:,:,k] - T3[:,:,k] + T4[:,:,k] - T5[:,:,k] - T6[:,:,k])/ \
                                    np.max(np.abs(np.array([T1[:,:,k], T2[:,:,k], T3[:,:,k], T4[:,:,k], T5[:,:,k], T6[:,:,k]])))
            ave_mesh_error = np.sum(self.Errors.mesh_error) / np.size(self.Errors.mesh_error)
            max_mesh_error = np.max(self.Errors.mesh_error)
            min_mesh_error = np.min(self.Errors.mesh_error[:,:,0:self.nx-1])

            # Moment Error
            for m in range(0, mtest):
                for k in range(0, self.nx - 1):
                    MT1 = np.sum(self.dvr_vol*(T1[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                    MT2 = np.sum(self.dvr_vol*(T2[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                    MT3 = np.sum(self.dvr_vol*(T3[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                    MT4 = np.sum(self.dvr_vol*(T4[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                    MT5 = np.sum(self.dvr_vol*(T5[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                    MT6 = np.sum(self.dvr_vol*(T6[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                    #NOTE This is correct for the original code, but is it correct mathematically?
                    moment_error[k,m] = np.abs(MT1 - MT2 - MT3 + MT4 - MT5 - MT6) / np.max(np.abs(np.array([MT1, MT2, MT3, MT4, MT5, MT6])))
                max_moment_error[m] = np.max(moment_error[:,m])

            # Compute error in qxH2_total
            # qxH2_total2 total neutral heat flux profile (watts m^-2)
            #    This is the total heat flux transported by the neutrals
            #    computed in a different way from:
            # 
            #    qxH2_total2(k)=vth3*total(Vr2pidVr*((vr2vx2(*,*,k)*fH2(*,*,k))#(Vx*dVx)))*0.5*(2*mu*mH)
            # 
            #    This should agree with qxH2_total if the definitions of nH2, pH2, piH2_xx,
            #    TH2, VxH2, and qxH2 are coded correctly.
            qxH2_total2 = np.zeros(self.nx)
            for k in range(0, self.nx):
                qxH2_total2[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**3)*np.sum(self.dvr_vol*((vr2vx2[:,:,k]*fH2[:,:,k]) @ (self.mesh.vx*self.dvx)))
            self.Errors.qxH2_total_error = np.abs(qxH2_total - qxH2_total2) / np.max(np.abs(np.array([qxH2_total, qxH2_total2])))

            # Compute error in QH2_total
            Q1 = np.zeros(self.nx)
            Q2 = np.zeros(self.nx)
            self.Errors.QH2_total_error = np.zeros(self.nx)
            for k in range(0, self.nx-1):
                Q1[k] = (qxH2_total[k+1] - qxH2_total[k]) / (self.mesh.x[k+1] - self.mesh.x[k])
                Q2[k] = 0.5*(QH2_total[k+1] + QH2_total[k])
            self.Errors.QH2_total_error = np.abs(Q1 - Q2)/np.max(np.abs(np.array([Q1, Q2])))

            if debrief > 0:
                print(self.prompt, 'Maximum particle convervation error of total collision operator: ', sval(np.max(self.Errors.C_Error)))
                print(self.prompt, 'Maximum H2_P_CX particle convervation error: ', sval(np.max(self.Errors.CX_Error)))
                print(self.prompt, 'Maximum H2_Wall particle convervation error: ', sval(np.max(Wall_error)))
                print(self.prompt, 'Maximum H2_H2_EL particle conservation error: ', sval(max_H2_H2_error[0]))
                print(self.prompt, 'Maximum H2_H2_EL x-momentum conservation error: ', sval(max_H2_H2_error[1]))
                print(self.prompt, 'Maximum H2_H2_EL total energy conservation error: ', sval(max_H2_H2_error[2]))
                print(self.prompt, 'Maximum H2_H_EL  particle conservation error: ', sval(max_H2_H_error[0]))
                print(self.prompt, 'Maximum H2_P_EL  particle conservation error: ', sval(max_H2_P_error[0]))
                print(self.prompt, 'Average mesh_error =', ave_mesh_error)
                print(self.prompt, 'Maximum mesh_error =', max_mesh_error)
                for m in range(0, 5):
                    print(self.prompt, 'Maximum fH2 vx^', sval(m), ' moment error: ', sval(max_moment_error[m]))
                print(self.prompt, 'Maximum qxH2_total error =', np.max(self.Errors.qxH2_total_error))
                print(self.prompt, 'Maximum QH2_total error =', np.max(self.Errors.QH2_total_error))
                if debug > 0:
                    input()

        fSH = np.zeros((self.nvr,self.nvx,self.nx))
        SH = np.zeros(self.nx)
        SP = np.zeros(self.nx)
        SHP = np.zeros(self.nx)
        ESH = np.zeros((self.nvr,self.nx))
        if compute_h_source:
            if debrief > 1:
                print(self.prompt, 'Computing Velocity Distributions of H products...')
            # Set Normalized Franck-Condon Velocity Distributions for reactions R2, R3, R4, R5, R6, R7, R8, R10

            # Make lookup table to select reaction Rn in SFCn
            #   Rn=2 3 4 5 6 7 8   10
            nFC = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 7])
            SFCn = np.zeros((self.nvr,self.nvx,self.nx,8))
            Eave = np.zeros((self.nx, 8))
            Emax = np.zeros((self.nx, 8))
            Emin = np.zeros((self.nx, 8))

            # Reaction R2: e + H2 -> e + H(1s) + H(1s)
            ii = nFC[2]
            Eave[:,ii] = 3.0 
            Emax[:,ii] = 4.25
            Emin[:,ii] = 2

            # Reaction R3: e + H2 -> e + H(1s) + H*(2s)
            ii = nFC[3]
            Eave[:,ii] = 0.3
            Emax[:,ii] = 0.55
            Emin[:,ii] = 0.0 

            # Reaction R4:  e + H2 -> e + H(+) + H(1s) + e
            ii = nFC[4]
            Ee = 3*self.mesh.Te/2     # Note the FC energy depends on electron energy
            kk = np.argwhere(Ee <= 26.0)
            if kk.size > 0:
                Eave[kk,ii] = 0.25
            kk = np.argwhere((Ee > 26.0) & (Ee <= 41.6))
            if kk.size > 0:
                Eave[kk,ii] = 0.5*(Ee[kk] - 26)
                Eave[kk,ii] = np.maximum(Eave[kk, ii], 0.25)
            kk = np.argwhere(Ee > 41.6)
            if kk.size > 0:
                Eave[kk,ii] = 7.8
            Emax[:,ii] = 1.5*Eave[:,ii]   # Note the max/min values here are a guess
            Emin[:,ii] = 0.5*Eave[:,ii]   # Note the max/min values here are a guess

            # Reaction R5: e + H2 -> e + H*(2p) + H*(2s)
            ii = nFC[5]
            Eave[:,ii] = 4.85 
            Emax[:,ii] = 5.85
            Emin[:,ii] = 2.85

            # Reaction R6: e + H2 -> e + H(1s) + H*(n=3)
            ii = nFC[6]
            Eave[:,ii] = 2.5 
            Emax[:,ii] = 3.75 
            Emin[:,ii] = 1.25 

            # Reaction R7: e + H2(+) -> e + H(+) + H(1s)
            ii = nFC[7]
            Eave[:,ii] = 4.3   
            Emax[:,ii] = 4.3 + 2.1     # Note the max/min values here are a guess
            Emin[:,ii] = 4.3 - 2.1     # Note the max/min values here are a guess

            # Reaction R8: e + H2(+) -> e + H(+) + H*(n=2)
            ii = nFC[8]
            Eave[:,ii] = 1.5 
            Emax[:,ii] = 1.5 + 0.75     # Note the max/min values here are a guess
            Emin[:,ii] = 1.5 - 0.75     # Note the max/min values here are a guess 

            # Reaction R10: e + H2(+) -> H(1s) + H*(n>=2)
            ii = nFC[10]

            # Compute relative cross-sections for populating a specific n level for reaction R10
            # (see page 62 in Janev, "Elementary Processes in Hydrogen-Helium Plasmas", Springer-Verlag, 1987)
            #   n=2   3    4    5    6
            R10rel = np.array([0.1, 0.45, 0.22, 0.12, 0.069])
            for k in range(7, 11): 
                R10rel = np.append(R10rel, 10/(k**3))
            En = 13.58/((2 + np.arange(9))**2) # Energy of Levels

            for k in range(0, self.nx):
                truncate_point = np.minimum(len(Ee), len(En))
                EHn = 0.5*(Ee[:truncate_point] - En[:truncate_point])*R10rel/np.sum(R10rel) #NOTE For some reason, this is how idl handles subtracting two arrays of different sizes
                EHn = np.maximum(EHn, 0)
                Eave[k,ii] = np.sum(EHn)
                Eave[k,ii] = np.maximum(Eave[k,ii], 0.25)
                Emax[k,ii] = 1.5*Eave[k,ii] # Note the max/min values here are a guess
                Emin[k,ii] = 0.5*Eave[k,ii] # Note the max/min values here are a guess
            
            # Set SFCn values for reactions R2, R3, R4, R5, R6, R7, R8, R10
            Vfc = np.zeros((self.nvr,self.nvx,self.nx))
            Tfc = np.zeros((self.nvr,self.nvx,self.nx))
            magV = np.sqrt(vr2vx2)
            _THP = np.zeros((self.nvr,self.nvx,self.nx))
            _TH2 = np.zeros((self.nvr,self.nvx,self.nx)) 
            for k in range(0, self.nx):
                _THP[:,:,k] = THP[k]/self.mesh.Tnorm
                _TH2[:,:,k] = self.H_Moments.TH[k]/self.mesh.Tnorm 
            
            # The following function is choosen to represent the velocity distribution of the
            # hydrogen products for a given reaction, accounting for the Franck-Condon energy
            # distribution and accounting for additional velocity spreading due to the finite
            # temperature of the molcules (neutral and ionic) prior to breakup:
            # 
            #     f(Vr,Vx) = exp( -0.5*mH*mu*(|v|-Vfc+0.5*Tfc/Vfc)^2/(Tfc+0.5*Tmol) )

            #       	|v|=sqrt(Vr^2+Vx^2)
            #	        Tfc= Franck-Condon 'temperature' = (Emax-Emin)/4
            #	        Vfc= Franck-Condon  velocity = sqrt(2 Eave/mH/mu)
            #		    Tmol= temperature of H2 molecule (neutral or ionic)

            #    This function is isotropic in velocity space and can be written in terms
            #  of a distribution in particle speed, |v|, 

            #     f(|v|) = exp( -(|v|-Vfc+1.5*Tfc/Vfc)^2/(Tfc+0.5*Tmol) )
            #
            # with velocities normalized by vth and T normalized by Tnorm.

            #  Recognizing the the distribution in energy, f(E), and particle speed, f(|v|),
            #  are related by  f(E) dE = f(|v|) 2 pi v^2 dv, and that dE/dv = mH mu v,
            #  f(E) can be written as

            #     f(E) = f(|v|) 2 pi |v|/(mH mu) = const. |v| exp( -(|v|-Vfc+1.5*Tfc/Vfc)^2/(Tfc+0.5*Tmol) )

            # The function f(Vr,Vx) was chosen because it has has the following characteristics:

            # (1) For Tmol/2 << Tfc,  the peak in the v^2 times the energy distribution, can be found
            #    by finding the |v| where df(E)/d|v| =0

            #    df(E)/d|v|= 0 = 3v^2 exp() - 2(|v|-Vfc+1.5*Tfc/Vfc)/Tfc v^3 exp() 
            #                    2(|v|-Vfc+1.5*Tfc/Vfc)/Tfc |v| = 3
            #    which is satisfied when |v|=Vfc. Thus the energy-weighted energy distribution peaks
            #    at the velocity corresponding to the average Franck-Condon energy.

            # (2) for Tmol/2 >> Tfc ~ Vfc^2, the velocity distribution becomes

            #	f(|v|) = exp( -2(|v|-Vfc+1.5*Tfc/Vfc)^2/Tmol )

            #    which leads to a velocity distribution that approaches the molecular velocity
            #    distribution with the magnitude of the average velocity divided by 2. This
            #    is the appropriate situation for when the Franck-Condon energies are negligible
            #    relative to the thermal speed of the molecules.

            Rn = np.array([2, 3, 4, 5, 6, 7, 8, 10])
            for jRn in range(np.size(Rn)):
                ii = nFC[Rn[jRn]]
                # print("ii", ii)
                Tfc[0,0,:] = 0.25*(Emax[:,ii] - Emin[:,ii])/self.mesh.Tnorm # Franck-Condon 'effective temperature'
                Vfc[0,0,:] = np.sqrt(Eave[:,ii]/self.mesh.Tnorm) # Velocity corresponding to Franck-Condon 'mean evergy'
                for k in range(self.nx):
                    Vfc[:,:,k] = Vfc[0,0,k]
                    Tfc[:,:,k] = Tfc[0,0,k]

                if Rn[jRn] <= 6:
                    # For R2-R6, the Franck-Condon 'mean energy' is taken equal to Eave
                    #	   and the 'temperature' corresponds to the sum of the Franck-Condon 'temperature', Tfc,
                    #          and the temperature of the H2 molecules, TH2. (Note: directed neutral molecule velocity
                    #	   is not included and assumed to be small)
                    arg = -(magV - Vfc + 1.5*Tfc/Vfc)**2 / (Tfc + 0.5*_TH2)
                    SFCn[:,:,:,ii] = np.exp(np.maximum(arg, -80))
                else: 
                #   For R7, R8 and R10, the Franck-Condon 'mean energy' is taken equal to Eave
                #	   and the 'temperature' corresponds to the sum of the Franck-Condon 'temperature', Tfc,
                #          and the temperature of the H2(+) molecular ions, THP. (Note: directed molecular ion velocity
                #	   is not included and assumed to be small)    
                    arg = -(magV - Vfc + 1.5*Tfc/Vfc)**2 / (Tfc + 0.5*_THP)
                    SFCn[:,:,:,ii] = np.exp(np.maximum(arg, -80))

                for k in range(self.nx):
                    SFCn[:,:,k,ii] = SFCn[:,:,k,ii] / (np.sum(self.dvr_vol*(SFCn[:,:,k,ii] @ self.dvx)))

            # NOTE Add Plotting back in later
            

            self.Errors.vbar_error = np.zeros(self.nx)
            if compute_errors:
                # Test: The average speed of a non-shifted maxwellian should be 2*Vth*sqrt(Ti(x)/Tnorm)/sqrt(!pi)
                TFC = np.min(Eave[0,:]) + ((np.max(Eave[0,:]) - np.min(Eave[0,:]))*np.arange(0, self.nx) / (self.nx - 1))
                vx_shift = np.zeros_like(TFC)
                Tmaxwell = TFC
                mol = 1
                Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
                vbar_test = self.vth*np.sqrt(vr2vx2[:,:,0])
                for k in range(0, self.nx):
                    vbar = np.sum(self.dvr_vol*((vbar_test*Maxwell[:,:,k]) @ self.dvx))
                    vbar_exact = 2*self.vth*np.sqrt(TFC[k] / self.mesh.Tnorm) / np.sqrt(np.pi)
                    self.Errors.vbar_error[k] = np.abs(vbar - vbar_exact) / vbar_exact
                if debrief > 0: 
                    print(self.prompt, 'Maximum Vbar error over FC energy range = ', np.max(self.Errors.vbar_error))

            # Compute atomic hydrogen source distribution function
            # using normalized FC source distributions SFCn
            fSH_calc = lambda k,x : sigv[k,x]*SFCn[:,:,k,nFC[x]]
            for k in range(0, self.nx):

                fSH[:,:,k] = self.mesh.ne[k]*self.H_Moments.nH[k]*(2*fSH_calc(k,2) + 2*fSH_calc(k,3) + fSH_calc(k,4) + 2*fSH_calc(k,5) + 2*fSH_calc(k,6))
                
                fSH[:,:,k] = fSH[:,:,k] + self.mesh.ne[k]*nHP[k]*(fSH_calc(k,7) + fSH_calc(k,8) + 2*fSH_calc(k,10))

            # Compute total H and H(+) sources
            for k in range(self.nx):
                SH[k] = np.sum(self.dvr_vol*(fSH[:,:,k] @ self.dvx))
                SP[k] = self.mesh.ne[k]*self.H_Moments.nH[k]*sigv[k,4] + self.mesh.ne[k]*nHP[k]*(sigv[k,7] + sigv[k,8] + 2*sigv[k,9])

            # Compute total HP source
            SHP = self.mesh.ne*self.H_Moments.nH*sigv[:,1]

            # Compute energy distrobution of H source 
            for k in range(0, self.nx):
                ESH[:,k] = (self.Eaxis*fSH[:,self.vx_pos[0],k]*self.dvr_vol_h_order) / self.dEaxis
                ESH[:,k] = ESH[:,k] / np.max(ESH[:,k])
            
            Source_Error = np.zeros(self.nx)

            # Compute Source Error
            if compute_errors:
                if debrief > 1:
                    print(self.prompt, 'Computing Source Error')
                # Test Mass Balance
                # The relationship, 2 dGammaxH2/dx - 2 SH2 + SH + SP + 2 nHp x Nuloss = 0, should be satisfied.
                dGammaxH2dx = np.zeros((self.nx-1))
                SH_p = np.zeros(self.nx-1)
                for k in range(0, self.nx-1):
                    dGammaxH2dx[k] = (GammaxH2[k+1] - GammaxH2[k]) / (self.mesh.x[k+1] - self.mesh.x[k])
                for k in range(0, self.nx-1):
                    SH_p[k] = 0.5*(SH[k+1] + SP[k+1] + 2*self.NuLoss[k+1]*nHP[k+1] - 2*SH2[k+1] + SH[k] + SP[k] + 2*self.NuLoss[k]*nHP[k] - 2*SH2[k])
                max_source = np.max(np.array([SH, 2*SH2]))
                for k in range(0, self.nx - 1):
                    Source_Error[k] = np.abs(2*dGammaxH2dx[k] + SH_p[k]) / np.max(np.abs(np.array([2*dGammaxH2dx[k], SH_p[k], max_source])))
                if debrief > 0:
                    print(self.prompt, 'Maximum Normalized Source_error =', np.max(Source_Error))
            
            
        # Set common blocks

        # Kinetic_H2_input common  
        self.Input.vx_s = self.mesh.vx
        self.Input.vr_s = self.mesh.vr
        self.Input.x_s = self.mesh.x
        self.Input.Tnorm_s = self.mesh.Tnorm
        self.Input.mu_s = self.mu
        self.Input.Ti_s = self.mesh.Ti
        self.Input.Te_s = self.mesh.Te
        self.Input.n_s = self.mesh.ne
        self.Input.vxi_s = self.vxi
        self.Input.fH2BC_s = self.fH2BC
        self.Input.GammaxH2BC_s = self.GammaxH2BC
        self.Input.NuLoss_s = self.NuLoss
        self.Input.PipeDia_s = self.mesh.PipeDia
        self.Input.fH_s = fH
        self.Input.SH2_s = SH2
        self.Input.fH2_s = fH2

        self.Input.nHP_s = nHP
        self.Input.THP_s = THP
        self.Input.Simple_CX_s = self.COLLISIONS.Simple_CX
        self.Input.Sawada_s = sawada
        self.Input.H2_H2_EL_s = self.COLLISIONS.H2_H2_EL
        self.Input.H2_P_EL_s = self.COLLISIONS.H2_P_EL
        self.Input.H2_H_EL_s = self.COLLISIONS.H2_H_EL
        self.Input.H2_HP_CX_s = self.COLLISIONS.H2_P_CX
        self.Input.ni_correct_s = ni_correct

        # kinetic_h2_internal common block  
        self.Internal.vr2vx2 = vr2vx2
        self.Internal.vr2vx_vxi2 = vr2vx_vxi2
        self.Internal.fw_hat = fw_hat
        self.Internal.fi_hat = fi_hat
        self.Internal.fHp_hat = fHp_hat
        self.Internal.EH2_P = EH2_P
        self.Internal.sigv = sigv
        self.Internal.Alpha_Loss = Alpha_Loss
        self.Internal.v_v2 = v_v2
        self.Internal.v_v = v_v
        self.Internal.vr2_vx2 = vr2_vx2
        self.Internal.vx_vx = vx_vx

        self.Internal.Vr2pidVrdVx = Vr2pidVrdVx
        self.Internal.SIG_CX = SIG_CX
        self.Internal.SIG_H2_H2 = SIG_H2_H2
        self.Internal.SIG_H2_H = SIG_H2_H
        self.Internal.SIG_H2_P = SIG_H2_P
        self.Internal.Alpha_CX = Alpha_CX
        self.Internal.Alpha_H2_H = Alpha_H2_H
        self.Internal.MH2_H2_sum = MH2_H2_sum
        self.Internal.Delta_nH2s = Delta_nH2s

        
        if debug > 0:
            print(self.prompt, 'Finished')
        return (fH2, nHP, THP, self.H_Moments.nH, GammaxH2, self.H_Moments.VxH, pH2, self.H_Moments.TH, qxH2, qxH2_total, Sloss, QH2, RxH2, QH2_total, AlbedoH2, \
            WallH2, fSH, SH, SP, SHP, NuE, NuDis, ESH, self.Eaxis)



    # ------ Testing Functions ------

    def _test_init_parameters(self):
        '''
        Performs compatibility tests for passed parameters when initializing class
        '''

        dx = self.mesh.x - np.roll(self.mesh.x, 1)
        dx = dx[1:]
        notpos = np.argwhere(dx <= 0.0)
        if notpos.size > 0:
            raise Exception(self.prompt + " x must be increasing with index!")
        if (self.nvx % 2) != 0:
            raise Exception(self.prompt + " Number of elements in vx must be even!")
        if self.mesh.Ti.size != self.nx:
            raise Exception(self.prompt + " Number of elements in Ti and x do not agree!")
        if self.vxi is None:
            self.vxi = np.zeros(self.nx)
        if np.size(self.vxi) != self.nx:
            raise Exception(self.prompt + " Number of elements in vxi and x do not agree!")
        if np.size(self.mesh.Te) != self.nx:
            raise Exception(self.prompt + " Number of elements in Te and x do not agree!")
        if np.size(self.mesh.ne) != self.nx:
            raise Exception(self.prompt + " Number of elements in n and x do not agree!")
        if self.NuLoss is None:
            self.NuLoss = np.zeros(self.nx)
        if np.size(self.NuLoss) != self.nx:
            raise Exception(self.prompt + " Number of elements in NuLoss and x do not agree!")
        if self.mesh.PipeDia is None:
            self.mesh.PipeDia = np.zeros(self.nx)
        if np.size(self.mesh.PipeDia) != self.nx:
            raise Exception(self.prompt + " Number of elements in PipeDia and x do not agree!")
        if self.GammaxH2BC is None:
            raise Exception(self.prompt + " GammaxH2BC is not defined!")
        if len(self.fH2BC[:,0]) != self.nvr:
            raise Exception(self.prompt + " Number of elements in fH2BC[:,0] and vr do not agree!")
        if len(self.fH2BC[0,:]) != self.nvx:
            raise Exception(self.prompt + " Number of elements in fH2BC[0,:] and vx do not agree!")
        count = np.size(np.argwhere(self.mesh.vr < 0))
        if count > 0:
            raise Exception(self.prompt + " vr contains zero or negative element(s)!")
        if np.sum(abs(self.mesh.x)) <= 0.0:
            raise Exception(self.prompt + " vx is all 0!")
        if np.sum(self.mesh.x) <= 0.0:
            raise Exception(self.prompt + " Total(x) is less than or equal to 0!")
        if self.mesh.Tnorm is None:
            raise Exception(self.prompt + " Tnorm is not defined!")
        if self.mu is None:
            raise Exception(self.prompt + " mu is not defined!")
        if self.mu != 1 and self.mu != 2:
            raise Exception(self.prompt + " mu must be 1 or 2!")
        if np.sum(abs(self.mesh.vr)) <= 0.0:
            raise Exception(self.prompt + " vr is all 0!")
        
        if np.size(self.vx_neg) < 1:
            raise Exception(self.prompt + " vx contains no negative elements!")
        if np.size(self.vx_pos) < 1:
            raise Exception(self.prompt + " vx contains no positive elements!")
        if np.size(self.vx_zero) > 0:
            raise Exception(self.prompt + " vx contains one or more zero elements!")
        diff = np.argwhere(self.mesh.vx[self.vx_pos] != -np.flipud(self.mesh.vx[self.vx_neg]))
        if diff.size > 0:
            raise Exception(self.prompt + " vx array elements are not symmetric about zero!")
        
        if np.sum(self.fH2BC_input) <= 0.0:
            raise Exception(self.prompt + " Values for fH2BC(:,:) with vx > 0 are all zero!")

        return
    

    def _test_input_parameters(self, fH, fH2, SH2, nHP, THP):
        '''
        Performs compatibility tests for passed parameters when calling run_generation
        '''

        if len(fH[:,0,0]) != self.nvr:
            raise Exception(self.prompt + " Number of elements in fH[:,0,0] and vr do not agree!")
        if len(fH[0,:,0]) != self.nvx:
            raise Exception(self.prompt + " Number of elements in fH[0,:,0] and vx do not agree!")
        if len(fH[0,0,:]) != self.nx:
            raise Exception(self.prompt + " Number of elements in fH[0,0,:] and x do not agree!")
        if len(fH2[:,0,0]) != self.nvr:
            raise Exception(self.prompt + " Number of elements in fH2[:,0,0] and vr do not agree!")
        if len(fH2[0,:,0]) != self.nvx:
            raise Exception(self.prompt + " Number of elements in fH2[0,:,0] and vx do not agree!")
        if len(fH2[0,0,:]) != self.nx:
            raise Exception(self.prompt + " Number of elements in fH2[0,0,:] and x do not agree!")
        if np.size(SH2) != self.nx:
            raise Exception(self.prompt + " Number of elements in SH2 and x do not agree!")
        if np.size(nHP) != self.nx:
            raise Exception(self.prompt + " Number of elements in nHP and x do not agree!")
        if np.size(THP) != self.nx:
            raise Exception(self.prompt + " Number of elements in THP and x do not agree!")