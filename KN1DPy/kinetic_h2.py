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

@dataclass
class MeshEqCoefficients():
    '''
    Mesh Equation values used in kinetic_h iteration
    Eqs. (3.22), (3.25), (3.30), (3.33)
    '''
    A: NDArray
    B: NDArray
    C: NDArray
    D: NDArray
    F: NDArray
    G: NDArray

@dataclass
class CollisionType():
    '''
    Data class for grouping H_H, H_P, and H_H2 elastic collision data
    '''
    H2_H2: NDArray
    H2_P: NDArray
    H2_H: NDArray

@dataclass
class KH2Results():
    '''
    Variables for results of KineticH2.run_procedure()
    See run_procedure for more detail on individual variables
    '''
    fH2: NDArray = None
    nHP: NDArray = None
    THP: NDArray = None
    nH2: NDArray = None
    GammaxH2: NDArray = None
    VxH2: NDArray = None
    pH2: NDArray = None
    TH2: NDArray = None
    qxH2: NDArray = None
    qxH2_total: NDArray = None
    Sloss: NDArray = None
    QH2: NDArray = None
    RxH2: NDArray = None
    QH2_total: NDArray = None
    AlbedoH2: float = None
    WallH2: NDArray = None
    fSH: NDArray = None
    SH: NDArray = None
    SP: NDArray = None
    SHP: NDArray = None
    NuE: NDArray = None
    NuDis: NDArray = None
    ESH: NDArray = None
    Eaxis: NDArray = None

class KineticH2():

    '''
    This class is part of the "KN1D" atomic and molecular neutal transport code.

    This class contains the data and methods to solve a 1-D spatial, 2-D velocity kinetic neutral transport
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

    # Internal Debug switches
    CI_Test = True
    Do_Alpha_CX_Test = False

    # Internal Tolerances 
    DeltaVx_tol = .01
    Wpp_tol = .001

    # Theta-prime Coordinate
    ntheta = 5 # use 5 theta mesh points for theta integration
    dtheta = np.ones(ntheta)/ntheta
    cos_theta = np.cos(np.pi*(np.arange(ntheta) + 0.5) / ntheta)

    # Internal Print Formatting
    prompt = 'Kinetic_H2 => '


    def __init__(self, mesh: KineticMesh, mu: int, vxi: NDArray, fH2BC: NDArray, GammaxH2BC: float, NuLoss: NDArray, SH2_initial: NDArray,
                    sawada: bool = True, compute_h_source: bool = False, ni_correct: bool = False, truncate: float = 1.0e-4, max_gen: int = 50, 
                    compute_errors: bool = False, debrief: int = 0, debug: int = 0):
        '''
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
            SH2_inital : ndarray, defualt=None
                Initial guess for source profile of wall-temperature H2 molecules (m^-3 s^-1). If None, zero array is used
            sawada : bool, default=False
                If false, disable Sawada correction
            compute_h_source : bool, default=False
                If true, compute fSH, SH, SP, and SHP
            ni_correct : bool, default=False
                If true, Corrects hydrogen ion density according to quasineutrality: ni=ne-nHp
            truncate : float, default=1.0e-4
                Convergence threshold for generations
            max_gen : int, default=50
                Max number of generations
            compute_errors : bool, default=False
                If true, compute error estimates
            debrief : int, default=0
                - 0=do not print
                - 1=print summary information
                - 2=print detailed information
            debug : int, default=0
                - 0=do not execute debug code
                - 1=summary debug
                - 2=detail debug
                - 3=very detailed debug
        '''

        # Configuration Options
        self.config = get_config()

        col = self.config['collisions']
        self.COLLISIONS = KH2Collisions(col['H2_H_EL'], col['H2_H2_EL'], col['H2_P_EL'], col['H2_P_CX'], col['SIMPLE_CX'])

        # Run Settings
        self.sawada = sawada
        self.compute_h_source = compute_h_source
        self.ni_correct = ni_correct
        self.truncate = truncate
        self.max_gen = max_gen
        self.compute_errors = (compute_errors and debrief)
        self.debrief = debrief
        self.debug = debug

        # Override settings for debug
        if self.debug > 0:
            self.debrief = np.maximum(self.debrief, 1)

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

        self.vth = np.sqrt(2 * CONST.Q * self.mesh.Tnorm / (self.mu * CONST.H_MASS))
        # Vr^2-2*Vx^2
        self.vr2_2vx2_2D = np.asarray([(vr**2) - 2*(self.mesh.vx**2) for vr in self.mesh.vr])
        # Differential Values
        differentials = VSpace_Differentials(self.mesh.vr, self.mesh.vx)
        self.dvr_vol = differentials.dvr_vol
        self.dvr_vol_h_order = differentials.dvr_vol_h_order
        self.dvx = differentials.dvx
        # FH2BC_Input
        self._init_fh2bc_input()
        # Determine Energy Space Differentials 
        self.Eaxis = (self.vth**2)*0.5*self.mu*CONST.H_MASS*(self.mesh.vr**2) / CONST.Q
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

        # Initial Computations
        # Some may not be used depending on inputs
        self._init_static_internals(SH2_initial)

        return


    def run_procedure(self, fH: NDArray = None, SH2: NDArray = None, fH2: NDArray = None, nHP: NDArray = None, THP: NDArray = None) -> KH2Results:
        '''
        Solves a 1-D spatial, 2-D velocity kinetic neutral transport 
        problem for molecular hydrogen or deuterium (H2)

        Parameters
        ----------
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
            
        Returns
        -------
        KH2Results

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
        '''


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
        if np.sum(fH) <= 0:
            self.COLLISIONS.H2_H_EL = False

        # Scale input molecular distribution function to agree with desired flux
        fH2[:,self.vx_pos,0] = self.fH2BC_input[:,self.vx_pos]


        # --- Compute Variables---

        Do_Alpha_CX, Do_Alpha_H2_P = self._compute_dynamic_internals(fH, fH2, nHP, THP)

        # Compute nH2
        nH2 = np.zeros((self.nx))
        for k in range(0, self.nx):
            nH2[k] = np.sum(self.dvr_vol*(np.matmul(fH2[:,:,k], self.dvx)))

        # Compute Side-Wall collision rate
        gamma_wall = np.zeros((self.nvr,self.nvx,self.nx))
        for k in range(0, self.nx):
            if self.mesh.PipeDia[k] > 0:
                gamma_wall[:,:,k] = 2*self.mesh.vr/self.mesh.PipeDia[k]

        
        # --- Iteration ---

        fH2, alpha_c, Beta_CX_sum, Swall_sum, collision_freqs, m_sums = self._run_iteration_scheme(fH2, nH2, nHP, THP, SH2, gamma_wall, Do_Alpha_CX, Do_Alpha_H2_P)


        # --- Compute results ---

        results = self._compile_results(fH2, SH2, gamma_wall, alpha_c, Swall_sum, Beta_CX_sum, collision_freqs, m_sums)

        if self.compute_errors:
            self._compute_final_errors(results, SH2, Beta_CX_sum, Swall_sum, m_sums, alpha_c, collision_freqs)
  

        # --- Compute H Source ---

        results = self._compute_h_source(results, SH2)
            
            
        # --- Save input Variables ---

        self.Input.fH_s = fH
        self.Input.SH2_s = SH2
        self.Input.fH2_s = results.fH2
        self.Input.nHP_s = results.nHP
        self.Input.THP_s = results.THP
        self.Input.ni_correct_s = self.ni_correct

        
        if self.debug > 0:
            print(self.prompt, 'Finished')

        return results



    # ------ Main Procedure Functions ------

    def _run_iteration_scheme(self, fH2, nH2, nHP, THP, SH2, gamma_wall, Do_Alpha_CX, Do_Alpha_H2_P):
        '''
        Implements the fH2 iteration, iterates until fH2 converges.
        Convergence is determined by evaluating changes in ion density, with iterations
        terminating when density change is low enough.
        '''
        # Set iteration Scheme 
        fH2_iterate = False
        if self.COLLISIONS.H2_H2_EL or self.COLLISIONS.H2_P_CX or self.COLLISIONS.H2_H_EL or self.COLLISIONS.H2_P_EL or self.ni_correct:
            fH2_iterate = True

        # Begin Iteration
        fH2G = np.zeros((self.nvr,self.nvx,self.nx))
        NH2G = np.zeros((self.nx, self.max_gen+1))
        while True:
            nH_input = copy.copy(nH2)

            # Compute Alpha_CX for present THP and nHP, if it is needed and has not
            # already been computed with the present parameters
            if Do_Alpha_CX:
                self._compute_alpha_cx(nHP, THP)

            # Compute Alpha_H2_P for present Ti and ni (optionally correcting for nHP), 
            # if it is needed and has not already been computed with the present parameter
            if Do_Alpha_H2_P:
                self._compute_alpha_h2_p(nHP)
            

            # --- Compute Collision Frequency ---

            # Omega Values (collision frequencies)
            collision_freqs = self._compute_omega_values(fH2, nH2)
            # total Collision Frequency
            alpha_c = self._compute_collision_frequency(collision_freqs, gamma_wall)
            
            # Generate Coefficients
            meq_coeffs = self._compute_mesh_equation_coefficients(alpha_c, SH2)


            # --- 0th Generation ---

            # Compute first-flight (0th generation) neutral distribution function
            if self.debrief > 0:
                print(self.prompt, 'Computing molecular neutral generation#', sval(0))
            fH2G[:,self.vx_pos,0] = fH2[:,self.vx_pos,0]
            for k in range(0, self.nx-1):
                fH2G[:,self.vx_pos,k+1] = fH2G[:,self.vx_pos,k]*meq_coeffs.A[:,self.vx_pos,k] + meq_coeffs.F[:,self.vx_pos,k]
            for k in range(self.nx-1,0,-1):
                fH2G[:,self.vx_neg,k-1] = fH2G[:,self.vx_neg,k]*meq_coeffs.C[:,self.vx_neg,k] + meq_coeffs.G[:,self.vx_neg,k]

            # Compute first-flight neutral density profile 
            for k in range(0, self.nx):
                NH2G[k, 0] = np.sum(self.dvr_vol*(fH2G[:,:,k] @ self.dvx))

            # Set total molecular neutral distrobution function to first flight generation 
            fH2 = copy.copy(fH2G)
            nH2 = NH2G[:,0]


            # --- Iterative Generations ---

            fH2, nH2, fH2G, NH2G, Swall_sum, Beta_CX_sum, m_sums, igen = self._run_generations(fH2, nH2, fH2G, NH2G, nHP, gamma_wall, meq_coeffs, collision_freqs, fH2_iterate)
            self.Internal.MH2_H2_sum = m_sums.H2_H2

            # Compute needed results for iteration
            nH2, _, _, _, _, _, _, _, nHP, THP = self._compute_iteration_results(fH2)
            

            # --- End Iteration ---

            if fH2_iterate:
                # Compute 'seed error': Delta_nH2s=(|nH2s-nH2|)/max(nH2) 
                # If Delta_nH2s is greater than 10*truncate then iterate fH2

                self.Internal.Delta_nH2s = np.max(np.abs(nH_input - nH2))/np.max(nH2)
                if self.Internal.Delta_nH2s <= 10*self.truncate:
                    break
        
        # --- Update Last Generation ---

        # Update Swall_sum using last generation
        Swall = self._compute_swall(fH2G, gamma_wall)
        Swall_sum += Swall

        # Update Beta_CX_sum using last generation
        Beta_CX = self._compute_beta_cx(fH2G, nHP)
        Beta_CX_sum += Beta_CX

        # Update MH2_*_sum using last generation
        m_vals = self._compute_mh_values(fH2G, NH2G[:,igen-1])
        m_sums.H2_H += m_vals.H2_H
        m_sums.H2_P += m_vals.H2_P
        self.Internal.MH2_H2_sum += m_vals.H2_H2

        return fH2, alpha_c, Beta_CX_sum, Swall_sum, collision_freqs, m_sums


    def _run_generations(self, fH2, nH2, fH2G, NH2G, nHP, gamma_wall, meq_coeffs, collision_freqs, fH2_iterate):
        '''
        Iterate through and compute generations of collision
        '''
        
        Swall_sum = np.zeros((self.nvr,self.nvx,self.nx))
        Beta_CX_sum = np.zeros((self.nvr,self.nvx,self.nx))
        m_sums = CollisionType(np.zeros((self.nvr,self.nvx,self.nx)), np.zeros((self.nvr,self.nvx,self.nx)), np.zeros((self.nvr,self.nvx,self.nx)))
        
        igen = 0
        while True:
            if igen+1 > self.max_gen or fH2_iterate == 0: 
                if self.debrief > 1:
                    print(self.prompt,'Completed ', sval(self.max_gen), ' generations. Returning present solution...')
                break
            igen = igen + 1
            if self.debrief > 0: 
                print(self.prompt, 'Computing molecular neutral generation#', sval(igen))
        
            #Compute Swall from previous generation
            Swall = self._compute_swall(fH2G, gamma_wall)
            #Sum wall collision source over all generations
            Swall_sum += Swall

            #Compute Beta_CX from previous generation
            Beta_CX = self._compute_beta_cx(fH2G, nHP)
            #Sum charge exchange source over all generations
            Beta_CX_sum += Beta_CX

            # Elastic collision maxwellians
            m_vals = self._compute_mh_values(fH2G, NH2G[:,igen-1])
            m_sums.H2_H += m_vals.H2_H
            m_sums.H2_P += m_vals.H2_P
            m_sums.H2_H2 += m_vals.H2_H2

            # Compute next generation molecular distribution
            OmegaM = collision_freqs.H2_H2*m_vals.H2_H2 + collision_freqs.H2_P*m_vals.H2_P + collision_freqs.H2_H*m_vals.H2_H
            fH2G[:] = 0.0
            for k in range(0, self.nx-1):
                fH2G[:,self.vx_pos,k+1] = meq_coeffs.A[:,self.vx_pos,k]*fH2G[:,self.vx_pos,k] + meq_coeffs.B[:,self.vx_pos,k]*(Swall[:,self.vx_pos,k+1] + Beta_CX[:,self.vx_pos,k+1] + OmegaM[:,self.vx_pos,k+1] + Swall[:,self.vx_pos,k] + Beta_CX[:,self.vx_pos,k] + OmegaM[:,self.vx_pos,k])
            for k in range(self.nx-1, 0, -1):
                fH2G[:,self.vx_neg,k-1] = meq_coeffs.C[:,self.vx_neg,k]*fH2G[:,self.vx_neg,k] + meq_coeffs.D[:,self.vx_neg,k]*(Swall[:,self.vx_neg,k-1] + Beta_CX[:,self.vx_neg,k-1] + OmegaM[:,self.vx_neg,k-1] + Swall[:,self.vx_neg,k] + Beta_CX[:,self.vx_neg,k] + OmegaM[:,self.vx_neg,k])
            for k in range(0, self.nx):
                NH2G[k,igen] = np.sum(self.dvr_vol*(fH2G[:,:,k] @ self.dvx))
            

            # Add result to total neutral distribution function
            fH2 = fH2 + fH2G
            nH2 = nH2 + NH2G[:,igen]

            # Compute 'generation error': Delta_nH2G=max(NH2G(*,igen)/max(nH2))
            # and decide if another generation should be computed
            Delta_nH2G = np.max(NH2G[:,igen]/np.max(nH2))
            if (Delta_nH2G < self.truncate) or (fH2_iterate and (Delta_nH2G < 0.003*self.Internal.Delta_nH2s)):
                # If fH2 'seed' is being iterated, then do another generation until the 'generation error'
                # is less than 0.003 times the 'seed error' or is less than TRUNCATE
                break
            # If fH2 'seed' is NOT being iterated, then do another generation unitl the 'generation error'
            # is less than parameter TRUNCATE

        return fH2, nH2, fH2G, NH2G, Swall_sum, Beta_CX_sum, m_sums, igen
    

    def _compute_iteration_results(self, fH2):
        '''
        Computes results of the iteration scheme
        '''

        # Compute H2 density profile
        nH2 = np.zeros(self.nx)
        for k in range(0, self.nx): #NOTE Should the rest of this be in the loop? KH, nH2 is only thing computed in the loop
            nH2[k] = np.sum(self.dvr_vol*(fH2[:,:,k] @ self.dvx))

        # GammaxH2 - particle flux in x direction
        GammaxH2 = np.zeros(self.nx)
        for k in range(0, self.nx):
            GammaxH2[k] = self.vth*np.sum(self.dvr_vol*(fH2[:,:,k] @ (self.mesh.vx*self.dvx)))

        # VxH2 - x velocity
        VxH2 = GammaxH2 / nH2
        _VxH2 = VxH2 / self.vth 

        # magnitude of random velocity at each mesh point 
        vr2vx2_ran = np.zeros((self.nvr, self.nvx, self.nx))
        for i in range(0, self.nvr):
            # vr2vx2_ran[i,:,:] = self.mesh.vr[i]**2 + (self.mesh.vx - VxH2/self.vth)**2
            for k in range(0, self.nx):
                vr2vx2_ran[i,:,k] = self.mesh.vr[i]**2 + (self.mesh.vx - _VxH2[k])**2

        # pH2 - pressure 
        pH2 = np.zeros(self.nx)
        for k in range(0, self.nx):
            pH2[k] = ((2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((vr2vx2_ran[:,:,k]*fH2[:,:,k]) @ self.dvx)))/(3*CONST.Q)

        #TH2 - temperature 
        TH2 = pH2/nH2

        # Compute NuDis - Dissociation frequency 
        NuDis = self.mesh.ne*np.sum(self.Internal.sigv[:,7:11], 1)
        
        # Compute NuE (assume np=ne) - Energy equilibration frequency H(+) <-> H2(+)
        NuE = (7.7e-7*self.mesh.ne*1.0e-6)/(np.sqrt(self.mu)*(self.mesh.Ti**1.5))
        
        # Compute H2(+) density profile
        nHP = (nH2*self.mesh.ne*self.Internal.sigv[:,1])/(NuDis + self.NuLoss)

        # Compute THP - temperature of molecular ions
        THP = (self.mesh.Ti*NuE)/(NuE + NuDis + self.NuLoss)

        return nH2, GammaxH2, VxH2, vr2vx2_ran, pH2, TH2, NuDis, NuE, nHP, THP
    

    def _compile_results(self, fH2, SH2, gamma_wall, alpha_c, Swall_sum, Beta_CX_sum, collision_freqs, m_sums) -> KH2Results:
        '''
        Computes final results of kinetic_h2 procedure, compiles into KH2Results dataclass
        '''

        # --- Recompute Results from end of iteration ---

        nH2, GammaxH2, VxH2, vr2vx2_ran, pH2, TH2, NuDis, NuE, nHP, THP = self._compute_iteration_results(fH2)
        

        # --- Compute remaining moments ---

        _VxH2 = VxH2 / self.vth

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
            C = self.vth*((self.Internal.fw_hat[:,:]*SH2[k]/self.vth) + Swall_sum[:,:,k] + Beta_CX_sum[:,:,k] - (alpha_c[:,:,k]*fH2[:,:,k]) + \
                    (collision_freqs.H2_P[k]*m_sums.H2_P[:,:,k]) + (collision_freqs.H2_H[k]*m_sums.H2_H[:,:,k]) + (collision_freqs.H2_H2[k]*self.Internal.MH2_H2_sum[:,:,k]))
            QH2[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((vr2vx2_ran[:,:,k]*C) @ self.dvx))
            RxH2[k] = (2*self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_vol*(C @ (self.dvx*(self.mesh.vx - _VxH2[k]))))
            Sloss[k] = -np.sum(self.dvr_vol*(C @ self.dvx)) + SH2[k]
            WallH2[k] = np.sum(self.dvr_vol*((gamma_wall[:,:,k]*fH2[:,:,k]) @ self.dvx))

            if self.COLLISIONS.H2_H_EL:
                CH2_H = self.vth*collision_freqs.H2_H[k]*(m_sums.H2_H[:,:,k] - fH2[:,:,k])
                self.Output.RxH_H2[k] = (2*self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_vol*(CH2_H @ (self.dvx*(self.mesh.vx - _VxH2[k]))))
                self.Output.EH_H2[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((self.Internal.vr2vx2[:,:,k]*CH2_H) @ self.dvx))

            if self.COLLISIONS.H2_P_EL:
                CH2_P = self.vth*collision_freqs.H2_P[k]*(m_sums.H2_P[:,:,k] - fH2[:,:,k])
                self.Output.RxP_H2[k] = (2*self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_vol*(CH2_P @ (self.dvx*(self.mesh.vx - _VxH2[k]))))
                self.Output.EP_H2[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((self.Internal.vr2vx2[:,:,k]*CH2_P) @ self.dvx))

            if self.COLLISIONS.H2_P_CX:
                CH2_HP_CX = self.vth*(Beta_CX_sum[:,:,k] - self.Internal.Alpha_CX[:,:,k]*fH2[:,:,k])
                self.Output.RxH2CX[k] = (2*self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_vol*(CH2_HP_CX @ (self.dvx*(self.mesh.vx - _VxH2[k]))))
                self.Output.EH2CX[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((self.Internal.vr2vx2[:,:,k]*CH2_HP_CX) @ self.dvx))

            CW_H2 = self.vth*(Swall_sum[:,:,k] - gamma_wall[:,:,k]*fH2[:,:,k])
            self.Output.RxW_H2[k] = (2*self.mu*CONST.H_MASS)*self.vth*np.sum(self.dvr_vol*(CW_H2 @ (self.dvx*(self.mesh.vx - _VxH2[k]))))
            self.Output.EW_H2[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((self.Internal.vr2vx2[:,:,k]*CW_H2) @ self.dvx))

            if self.COLLISIONS.H2_H2_EL:
                vr2_2vx_ran2 = np.zeros((self.nvr,self.nvx))
                CH2_H2 = self.vth*collision_freqs.H2_H2[k]*(self.Internal.MH2_H2_sum[:,:,k] - fH2[:,:,k])
                for i in range(0, self.nvr):
                    vr2_2vx_ran2[i,:] = self.mesh.vr[i]**2 - 2*((self.mesh.vx - _VxH2[k])**2)
                self.Output.Epara_PerpH2_H2[k] = -0.5*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((vr2_2vx_ran2*CH2_H2) @ self.dvx))

        # qxH2_total
        qxH2_total = (0.5*nH2*(2*self.mu*CONST.H_MASS)*VxH2*VxH2 + 2.5*pH2*CONST.Q)*VxH2 + CONST.Q*self.Output.piH2_xx*VxH2 + qxH2

        # QH2_total
        QH2_total = QH2 + RxH2*VxH2 - 0.5*(2*self.mu*CONST.H_MASS)*(Sloss - SH2)*VxH2*VxH2

        # Albedo
        gammax_plus = self.vth*np.sum(self.dvr_vol*(fH2[:,self.vx_pos,0] @ (self.mesh.vx[self.vx_pos]*self.dvx[self.vx_pos]))) 
        gammax_minus = self.vth*np.sum(self.dvr_vol*(fH2[:,self.vx_neg,0] @ (self.mesh.vx[self.vx_neg]*self.dvx[self.vx_neg])))
        AlbedoH2 = 0.0
        if np.abs(gammax_plus) > 0:
            AlbedoH2 = -gammax_minus/gammax_plus

        results = KH2Results(fH2, nHP, THP, nH2, GammaxH2, VxH2, pH2, TH2, qxH2, qxH2_total, Sloss, QH2, RxH2, QH2_total, AlbedoH2, WallH2, None, None, None, None, NuE, NuDis, None, self.Eaxis)

        return results
    


    # ------ H Source Functions ------

    def _compute_h_source(self, results, SH2) -> KH2Results:

        '''
        Set Normalized Franck-Condon Velocity Distributions for reactions R2, R3, R4, R5, R6, R7, R8, R10
        See Sections 2.1.3 (D, E, F, G, H)

        The following function is chosen to represent the velocity distribution of the
        hydrogen products for a given reaction, accounting for the Franck-Condon energy
        distribution and accounting for additional velocity spreading due to the finite
        temperature of the molcules (neutral and ionic) prior to breakup:
        
            f(Vr,Vx) = exp( -0.5*mH*mu*(|v|-Vfc+0.5*Tfc/Vfc)^2/(Tfc+0.5*Tmol) )

              	|v|=sqrt(Vr^2+Vx^2)
        	        Tfc= Franck-Condon 'temperature' = (Emax-Emin)/4
        	        Vfc= Franck-Condon  velocity = sqrt(2 Eave/mH/mu)
        		    Tmol= temperature of H2 molecule (neutral or ionic)

        This function is isotropic in velocity space and can be written in terms
        of a distribution in particle speed, |v|, 

            f(|v|) = exp( -(|v|-Vfc+1.5*Tfc/Vfc)^2/(Tfc+0.5*Tmol) )
        
        with velocities normalized by vth and T normalized by Tnorm.

        Recognizing the the distribution in energy, f(E), and particle speed, f(|v|),
        are related by  f(E) dE = f(|v|) 2 pi v^2 dv, and that dE/dv = mH mu v,
        f(E) can be written as

            f(E) = f(|v|) 2 pi |v|/(mH mu) = const. |v| exp( -(|v|-Vfc+1.5*Tfc/Vfc)^2/(Tfc+0.5*Tmol) )

        The function f(Vr,Vx) was chosen because it has has the following characteristics:

        (1) For Tmol/2 << Tfc,  the peak in the v^2 times the energy distribution, can be found
        by finding the |v| where df(E)/d|v| =0

           df(E)/d|v|= 0 = 3v^2 exp() - 2(|v|-Vfc+1.5*Tfc/Vfc)/Tfc v^3 exp() 
                           2(|v|-Vfc+1.5*Tfc/Vfc)/Tfc |v| = 3
        
        which is satisfied when |v|=Vfc. Thus the energy-weighted energy distribution peaks
        at the velocity corresponding to the average Franck-Condon energy.

        (2) for Tmol/2 >> Tfc ~ Vfc^2, the velocity distribution becomes

        	f(|v|) = exp( -2(|v|-Vfc+1.5*Tfc/Vfc)^2/Tmol )

        which leads to a velocity distribution that approaches the molecular velocity
        distribution with the magnitude of the average velocity divided by 2. This
        is the appropriate situation for when the Franck-Condon energies are negligible
        relative to the thermal speed of the molecules.
        '''

        results.fSH = np.zeros((self.nvr,self.nvx,self.nx))
        results.SH = np.zeros(self.nx)
        results.SP = np.zeros(self.nx)
        results.SHP = np.zeros(self.nx)
        results.ESH = np.zeros((self.nvr,self.nx))

        if not self.compute_h_source:
            # Stop Computation, return 0 arrays
            return results
        
        if self.debrief > 1:
            print(self.prompt, 'Computing Velocity Distributions of H products...')

        # Set SFCn values for reactions R2, R3, R4, R5, R6, R7, R8, R10
        SFCn = np.zeros((self.nvr,self.nvx,self.nx,8))
        Vfc = np.zeros((self.nvr,self.nvx,self.nx))
        Tfc = np.zeros((self.nvr,self.nvx,self.nx))
        magV = np.sqrt(self.Internal.vr2vx2)

        _THP = results.THP/self.mesh.Tnorm
        _TH2 = results.TH2/self.mesh.Tnorm 
        
        # Generate Lookup Table
        nFC, Eave, Emax, Emin = self._generate_h_source_table()

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
        

        self.Errors.vbar_error = np.zeros(self.nx)
        if self.compute_errors:
            # Test: The average speed of a non-shifted maxwellian should be 2*Vth*sqrt(Ti(x)/Tnorm)/sqrt(!pi)
            TFC = np.min(Eave[0,:]) + ((np.max(Eave[0,:]) - np.min(Eave[0,:]))*np.arange(0, self.nx) / (self.nx - 1))
            vx_shift = np.zeros_like(TFC)
            Tmaxwell = TFC
            mol = 1
            Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
            vbar_test = self.vth*np.sqrt(self.Internal.vr2vx2[:,:,0])
            for k in range(0, self.nx):
                vbar = np.sum(self.dvr_vol*((vbar_test*Maxwell[:,:,k]) @ self.dvx))
                vbar_exact = 2*self.vth*np.sqrt(TFC[k] / self.mesh.Tnorm) / np.sqrt(np.pi)
                self.Errors.vbar_error[k] = np.abs(vbar - vbar_exact) / vbar_exact
            if self.debrief > 0: 
                print(self.prompt, 'Maximum Vbar error over FC energy range = ', np.max(self.Errors.vbar_error))

        # Compute atomic hydrogen source distribution function
        # using normalized FC source distributions SFCn
        fSH_calc = lambda k,x : self.Internal.sigv[k,x]*SFCn[:,:,k,nFC[x]]
        for k in range(0, self.nx):

            results.fSH[:,:,k] = self.mesh.ne[k]*results.nH2[k]*(2*fSH_calc(k,2) + 2*fSH_calc(k,3) + fSH_calc(k,4) + 2*fSH_calc(k,5) + 2*fSH_calc(k,6))
            
            results.fSH[:,:,k] = results.fSH[:,:,k] + self.mesh.ne[k]*results.nHP[k]*(fSH_calc(k,7) + fSH_calc(k,8) + 2*fSH_calc(k,10))

        # Compute total H and H(+) sources
        for k in range(self.nx):
            results.SH[k] = np.sum(self.dvr_vol*(results.fSH[:,:,k] @ self.dvx))
            results.SP[k] = self.mesh.ne[k]*results.nH2[k]*self.Internal.sigv[k,4] + self.mesh.ne[k]*results.nHP[k]*(self.Internal.sigv[k,7] + self.Internal.sigv[k,8] + 2*self.Internal.sigv[k,9])

        # Compute total HP source
        results.SHP = self.mesh.ne*results.nH2*self.Internal.sigv[:,1]

        # Compute energy distribution of H source 
        for k in range(0, self.nx):
            results.ESH[:,k] = (self.Eaxis*results.fSH[:,self.vx_pos[0],k]*self.dvr_vol_h_order) / self.dEaxis
            results.ESH[:,k] = results.ESH[:,k] / np.max(results.ESH[:,k])
        
        Source_Error = np.zeros(self.nx)

        # Compute Source Error
        if self.compute_errors:
            if self.debrief > 1:
                print(self.prompt, 'Computing Source Error')
            # Test Mass Balance
            # The relationship, 2 dGammaxH2/dx - 2 SH2 + SH + SP + 2 nHp x Nuloss = 0, should be satisfied.
            dGammaxH2dx = np.zeros((self.nx-1))
            SH_p = np.zeros(self.nx-1)
            for k in range(0, self.nx-1):
                dGammaxH2dx[k] = (results.GammaxH2[k+1] - results.GammaxH2[k]) / (self.mesh.x[k+1] - self.mesh.x[k])
            for k in range(0, self.nx-1):
                SH_p[k] = 0.5*(results.SH[k+1] + results.SP[k+1] + 2*self.NuLoss[k+1]*results.nHP[k+1] - 2*SH2[k+1] + results.SH[k] + results.SP[k] + 2*self.NuLoss[k]*results.nHP[k] - 2*SH2[k])
            max_source = np.max(np.array([results.SH, 2*SH2]))
            for k in range(0, self.nx - 1):
                Source_Error[k] = np.abs(2*dGammaxH2dx[k] + SH_p[k]) / np.max(np.abs(np.array([2*dGammaxH2dx[k], SH_p[k], max_source])))
            if self.debrief > 0:
                print(self.prompt, 'Maximum Normalized Source_error =', np.max(Source_Error))

        return results
    

    def _generate_h_source_table(self):
        '''
        Create lookup table to select reaction Rn in SFCn, used for h_source computation
        '''
        #   Rn=2 3 4 5 6 7 8   10
        nFC = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 7])
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

        truncate_point = np.minimum(len(Ee), len(En))
        EHn = 0.5*(Ee[:truncate_point] - En[:truncate_point])*R10rel/np.sum(R10rel)
        EHn = np.maximum(EHn, 0)

        Eave[:,ii] = np.maximum(np.sum(EHn), 0.25)
        Emax[:,ii] = 1.5*Eave[:,ii] # Note the max/min values here are a guess
        Emin[:,ii] = 0.5*Eave[:,ii] # Note the max/min values here are a guess

        return nFC, Eave, Emax, Emin



    # ------ Variable Functions ------

    # --- init ---
        
    
    def _init_fh2bc_input(self):
        '''
        Computes fH2BC_input, used to scale molecular distribution function (fH2) to desired flux
        '''

        self.fH2BC_input = np.zeros(self.fH2BC.shape)
        self.fH2BC_input[:,self.vx_pos] = self.fH2BC[:,self.vx_pos]
        gamma_input = 1.0
        if abs(self.GammaxH2BC) > 0:
            gamma_input = self.vth*np.sum(self.dvr_vol*(self.fH2BC_input @ (self.mesh.vx*self.dvx)))
        ratio = abs(self.GammaxH2BC)/gamma_input
        self.fH2BC_input = self.fH2BC_input*ratio
        if abs(ratio - 1) > 0.01*self.truncate:
            self.fH2BC = self.fH2BC_input

        return
    

    def _init_static_internals(self, SH2_initial):
        '''
        Computes various internal variables based on constant mesh data
        '''

        self._init_grid(SH2_initial)
        self._init_protons()
        self._init_sigv()
        self._init_v_v2()
        self._init_sig_cx()
        self._init_sig_h_h2()
        self._init_sig_h2_p()
        self._init_sig_h2_h2()

        return


    def _init_grid(self, SH2_initial):
        '''
        Computes internal vr2vx2, vr2vx_vxi2, EH2_P, and fw_hat
        '''
        if self.debrief > 1:
            print(self.prompt, 'Computing vr2vx2, vr2vx_vxi2, EH2_P')
    
        # Magnitude of total normalized v^2 at each mesh point
        self.Internal.vr2vx2 = np.zeros((self.nvr,self.nvx,self.nx))
        for i in range(0, self.nvr):
            for k in range(0, self.nx):
                self.Internal.vr2vx2[i,:,k] = self.mesh.vr[i]**2 + self.mesh.vx**2

        # Magnitude of total normalized (v-vxi)^2 at each mesh point
        self.Internal.vr2vx_vxi2 = np.zeros((self.nvr,self.nvx,self.nx))
        for i in range(0, self.nvr):
            for k in range(0, self.nx):
                self.Internal.vr2vx_vxi2[i,:,k] = self.mesh.vr[i]**2 + (self.mesh.vx - self.vxi[k]/self.vth)**2

        # Molecular hydrogen ion energy in local rest frame of plasma at each mesh point
        EH2_P = CONST.H_MASS*self.Internal.vr2vx_vxi2*(self.vth**2) / CONST.Q
        EH2_P = np.maximum(EH2_P, 0.1)      # sigmav_cx does not handle neutral energies below 0.1 eV
        self.Internal.EH2_P = np.minimum(EH2_P, 2.0e4)    # sigmav_cx does not handle neutral energies above 20 keV

        # Compute Maxwellian H2 distribution at the wall temperature (fw_hat)
        self.Internal.fw_hat = np.zeros((self.nvr,self.nvx))
        if (np.sum(SH2_initial) > 0) | (np.sum(self.mesh.PipeDia) > 0):
            if self.debrief > 1:
                print(self.prompt, 'Computing fw_hat')
            vx_shift = np.array([0.0])
            Tmaxwell = np.array([CONST.TWALL])
            _maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,2,self.mesh.Tnorm)
            self.Internal.fw_hat = _maxwell[:,:,0]

        return
    

    def _init_protons(self):
        '''
        Computes internal fi_hat
        '''
        if self.debrief > 1:
            print(self.prompt, 'Computing fi_Hat')
        vx_shift = self.vxi
        Tmaxwell = self.mesh.Ti
        mol = 1
        Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
        self.Internal.fi_hat = np.copy(Maxwell)

        return
    

    def _init_sigv(self):
        '''
        Computes sigmav rates for each reaction and optionally applies
        CR model corrections of Sawada See (2.R1)-(2.R10).
        Also computes alpha_loss (Eq. 2.2)
        '''
        if self.debrief > 1:
            print(self.prompt, 'Computing sigv')

        sigv = np.zeros((self.nx,11))

        # Reaction R1:  e + H2 -> e + H2(+) + e 
        sigv[:,1] = sigmav_ion_hh(self.mesh.Te)
        if self.sawada:
            sigv[:,1] = sigv[:,1] * 3.7 / 2.0

        # Reaction R2:  e + H2 -> H(1s) + H(1s)
        sigv[:,2] = sigmav_h1s_h1s_hh(self.mesh.Te)
        if self.sawada:
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
        if self.sawada:
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

        self.Internal.sigv = sigv
        
        # Total H2 destruction rate (normalized by vth) = sum of reactions 1-6
        self.Internal.Alpha_Loss = self.mesh.ne*np.sum(sigv[:,1:7], axis = 1) / self.vth

        return
    

    def _init_v_v2(self):
        '''
        Compute v_prime values for charge exchange usage
        Computes v_v2, v_v, vr2_vx2, vx_vx, and vr2pidvrdvx
        '''
        # Set up arrays for charge exchange and elastic collision computations, if needed
        if self.debrief > 1:
            print(self.prompt, 'Computing v_v2, v_v, vr2_vx2, and vx_vx')
        # v_v2=(v-v_prime)^2 at each double velocity space mesh point, including theta angle
        self.Internal.v_v2 = np.zeros((self.nvr,self.nvx,self.nvr,self.nvx,self.ntheta))

        # vr2_vx2=(vr2 + vr2_prime - 2*vr*vr_prime*cos(theta) - 2*(vx-vx_prime)^2
        # at each double velocity space mesh point, including theta angle
        self.Internal.vr2_vx2 = np.zeros((self.nvr,self.nvx,self.nvr,self.nvx,self.ntheta))
        for m in range(0, self.ntheta): # double check this nested for loop 
            for l in range(0, self.nvx):
                for k in range(0, self.nvr):
                    for i in range(0, self.nvr):
                        v_starter = self.mesh.vr[i]**2 + self.mesh.vr[k]**2 - 2*self.mesh.vr[i]*self.mesh.vr[k]*self.cos_theta[m]
                        self.Internal.v_v2[i,:,k,l,m] = v_starter + (self.mesh.vx[:] - self.mesh.vx[l])**2
                        self.Internal.vr2_vx2[i,:,k,l,m] = v_starter - 2*(self.mesh.vx[:] - self.mesh.vx[l])**2
        # v_v=|v-v_prime| at each double velocity space mesh point, including theta angle
        self.Internal.v_v = np.sqrt(self.Internal.v_v2)

        # vx_vx=(vx-vx_prime) at each double velocity space mesh point
        self.Internal.vx_vx = np.zeros((self.nvr,self.nvx,self.nvr,self.nvx))
        for j in range(0,self.nvx):
            for l in range(0, self.nvx):
                self.Internal.vx_vx[:,j,:,l] = self.mesh.vx[j] - self.mesh.vx[l]

        # Set Vr'2pidVr'*dVx' for each double velocity space mesh point
        self.Internal.Vr2pidVrdVx = np.zeros((self.nvr,self.nvx,self.nvr,self.nvx))
        for k in range(0, self.nvr):
            self.Internal.Vr2pidVrdVx[:,:,k,:] = self.dvr_vol[k]
        for l in range(0, self.nvx):
            self.Internal.Vr2pidVrdVx[:,:,:,l] = self.Internal.Vr2pidVrdVx[:,:,:,l]*self.dvx[l]

        return
    

    def _init_sig_cx(self):
        '''
        Compute SigmaV_CX from sigma directly for present velocity space grid.
        Charge Exchange Option A
        '''
        if self.debrief > 1:
            print(self.prompt, 'Computing SIG_CX')

        # compute SIGMA_CX * v_v at all possible relative velocities
        _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx, self.ntheta))
        _Sig[:] = (self.Internal.v_v*sigma_cx_hh(self.Internal.v_v2*(CONST.H_MASS*(self.vth**2)/CONST.Q))).reshape(_Sig.shape, order='F')

        # Set SIG_CX = vr' x Integral{v_v*sigma_cx} over theta=0,2pi times differential velocity space element Vr'2pidVr'*dVx'
        SIG_CX = np.zeros((self.nvr*self.nvx, self.nvr*self.nvx))
        self.Internal.SIG_CX = (self.Internal.Vr2pidVrdVx*((_Sig @ self.dtheta).reshape(self.Internal.Vr2pidVrdVx.shape, order='F'))).reshape(SIG_CX.shape, order='F') 

        # SIG_CX is now vr' * sigma_cx(v_v) * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])

        return
    

    def _init_sig_h_h2(self):
        '''
        Compute SIG_H2_P for present velocity space grid
        '''
        if self.debrief > 1:
            print(self.prompt, 'Computing SIG_H2_P')

        # Compute sigma_H2_H * v_v at all possible relative velocities
        _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx, self.ntheta))
        _Sig[:] = (self.Internal.v_v*sigma_el_h_hh(self.Internal.v_v2*(0.5*CONST.H_MASS*(self.vth**2)/CONST.Q))).reshape(_Sig.shape, order='F')

        # Note: using H energy here for cross-section tabulated as H -> H2
        # Set SIG_H2_H = vr' x vx_vx x Integral{v_v * sigma_H2_H} over theta = 0, 2pi times differential velocity space element Vr'2pidVr'*dVx
        SIG_H2_H = np.zeros((self.nvr*self.nvx, self.nvr*self.nvx))
        self.Internal.SIG_H2_H = (self.Internal.Vr2pidVrdVx*self.Internal.vx_vx*((_Sig @ self.dtheta).reshape(self.Internal.vx_vx.shape, order='F'))).reshape(SIG_H2_H.shape, order='F')

        # SIG_H2_H is now vr' * vx_vx * sigma_H2_H(v_V) ( integrated over theta ) for all possible ([vr, vx], [vr', vx'])

        return
    

    def _init_sig_h2_p(self):
        '''
        Compute SIG_H2_P for present velocity space grid
        '''
        if self.debrief > 1:
            print(self.prompt, 'Computing SIG_H2_P')

        # Compute sigma_H2_P * v_v at all possible relative velocities
        _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx, self.ntheta))
        _Sig[:] = (self.Internal.v_v*sigma_el_p_hh(self.Internal.v_v2*(0.5*CONST.H_MASS*(self.vth**2)/CONST.Q))).reshape(_Sig.shape, order='F')
        #energy = v_v2*(0.5*CONST.H_MASS*Vth2/CONST.Q) #NOTE is this completely unused?

        # Note: using H energy here for cross-section tabulated as p -> H2

        # Set SIG_H2_P = vr' x vx_vx x Integral{v_v * sigma_H2_P} over theta = 0, 2pi times differential velocity space element Vr'2pidVr' * dVx
        SIG_H2_P = np.zeros((self.nvr*self.nvx, self.nvr*self.nvx))
        self.Internal.SIG_H2_P = (self.Internal.Vr2pidVrdVx*self.Internal.vx_vx*(_Sig @ self.dtheta).reshape(self.Internal.vx_vx.shape, order='F')).reshape(SIG_H2_P.shape, order='F')

        # SIG_H2_P is now vr' * vx_vx * sigma_h2_P(v_v) * v_v (integrated over theta) for all possible ([vr, vx], [vr', vx'])

        return


    def _init_sig_h2_h2(self):
        '''
        Compute SIG_H2_H2 for present velocity space grid
        '''
        if self.debrief > 1:
            print(self.prompt, 'Computing SIG_H2_H2')

        # Compute sigma_H2_H2 * vr2_vx2 * v_v at all possible relative velocities 
        _Sig = np.zeros((self.nvr*self.nvx*self.nvr*self.nvx, self.ntheta))
        _Sig[:] = (self.Internal.vr2_vx2*self.Internal.v_v*sigma_el_hh_hh(self.Internal.v_v2*(CONST.H_MASS*self.mu*(self.vth**2)/CONST.Q), vis = 1)/8.0).reshape(_Sig.shape, order='F')

        # Note : For viscosity, the cross section for D -> D is the same function of 
        # center of mass energy as H -> H.

        # Set SIG_H2_H2 = vr' x Integral{vr2_vx2*v_v*sigma_H2_H2} over theta=0,2pi times differential velocity space element Vr'2pidVr'*dVx'
        SIG_H2_H2 = np.zeros((self.nvr * self.nvx, self.nvr * self.nvx))
        self.Internal.SIG_H2_H2 = (self.Internal.Vr2pidVrdVx*((_Sig @ self.dtheta).reshape(self.Internal.Vr2pidVrdVx.shape, order='F'))).reshape(SIG_H2_H2.shape, order='F')

        # SIG_H2_H2 is now vr' * sigma_H2_H2(v_v) * vr2_vx2 * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])

        return
    

    # --- generational ---

    def _compute_dynamic_internals(self, fH, fH2, nHP, THP):
        '''
        Determines which internal variables need to be recomputed based on changes in input across iterations
        '''
        
        # Set flags to make use of previously computed local parameters
        New_fH = True
        if (self.Input.fH_s is not None) and np.array_equal(self.Input.fH_s, fH):
            New_fH = False

        New_H2_Seed = True
        if (self.Input.fH2_s is not None) and np.array_equal(self.Input.fH2_s, fH2):
            New_H2_Seed = False

        New_HP_Seed = True
        if (self.Input.nHP_s is not None) and np.array_equal(self.Input.nHP_s, nHP) and np.array_equal(self.Input.THP_s, THP):
            New_HP_Seed = False

        New_ni_correct = True
        if (self.Input.ni_correct_s is not None) and (self.Input.ni_correct_s != self.ni_correct):
            New_ni_correct = False 

        Do_fH_moments = (New_fH) and (np.sum(fH) > 0.0)
        Do_Alpha_CX =   ((self.Internal.Alpha_CX is None) | New_HP_Seed) & self.COLLISIONS.H2_P_CX
        Do_Alpha_H2_H = ((self.Internal.Alpha_H2_H is None) | New_fH) & self.COLLISIONS.H2_H_EL
        Do_Alpha_H2_P = ((self.Internal.Alpha_H2_P is None) | New_ni_correct) & self.COLLISIONS.H2_P_EL

        # Reset H Moments
        self.H_Moments.nH = np.zeros(self.nx)
        self.H_Moments.VxH = np.zeros(self.nx)
        self.H_Moments.TH = np.full(self.nx, 1.0)
        if Do_fH_moments:
            self._compute_fh_moments(fH)
            
        if Do_Alpha_H2_H == 1:
            self._compute_alpha_h_h2(fH)

        if New_H2_Seed:
            self.Internal.MH2_H2_sum = np.zeros((self.nvr,self.nvx,self.nx))
            self.Internal.Delta_nH2s = 1.0

        return Do_Alpha_CX, Do_Alpha_H2_P
    

    def _compute_fh_moments(self, fH):
        '''
        Computes moments from atomic hydrogen distribution functions
        '''
        if self.debrief > 1:
            print(self.prompt, 'Computing vx and T moments of fH')

        # Compute x flow velocity and temperature of atomic species
        vr2vx2_ran2 = np.zeros((self.nvr,self.nvx))
        for k in range(0, self.nx):
            self.H_Moments.nH[k] = np.sum(self.dvr_vol*(fH[:,:,k] @ self.dvx))
            if self.H_Moments.nH[k] > 0:
                self.H_Moments.VxH[k] = self.vth*np.sum(self.dvr_vol*(fH[:,:,k] @ (self.mesh.vx*self.dvx))) / self.H_Moments.nH[k]
                for i in range(0, self.nvr):
                    vr2vx2_ran2[i,:] = self.mesh.vr[i]**2 + (self.mesh.vx - self.H_Moments.VxH[k]/self.vth)**2
                self.H_Moments.TH[k] = (self.mu*CONST.H_MASS)*(self.vth**2)*np.sum((self.dvr_vol*((vr2vx2_ran2*fH[:,:,k]) @ self.dvx))) / (3*CONST.Q*self.H_Moments.nH[k])

        return
    

    def _compute_alpha_cx(self, nHP, THP):
        '''
        Compute charge exchange collision frequency (alpha_cx) using Eq.(2.10a) or (2.10b)
        '''
        if self.debrief > 1:
            print(self.prompt, 'Computing Alpha_CX')
        # Set Maxwellian Molecular Ion Distribution Function (assumed to be drifting with ion velocity, vxi)
        vx_shift = self.vxi
        Tmaxwell = THP
        mol = 2
        Maxwell = create_shifted_maxwellian(self.mesh.vr,self.mesh.vx,Tmaxwell,vx_shift,self.mu,mol,self.mesh.Tnorm)
        self.Internal.fHp_hat = copy.copy(Maxwell)

        if self.COLLISIONS.Simple_CX:
            # Option (B) : Use Maxwellian weighted <sigma v>
            
            # THP/mu at each mesh point
            THP_mu = np.zeros((self.nvr, self.nvx, self.nx))
            for k in range(0, self.nx):
                THP_mu[:,:,k] = THP[k]/self.mu

            # Molecular Charge Exchange sink rate 
            self.Internal.Alpha_CX = sigmav_cx_hh(THP_mu, self.Internal.EH2_P) / self.vth
            for k in range(0, self.nx):
                self.Internal.Alpha_CX[:,:,k] = self.Internal.Alpha_CX[:,:,k]*nHP[k]
        else:
            # Option (A): Compute SigmaV_CX from sigma directly via SIG_CX
            self.Internal.Alpha_CX = np.zeros((self.nvr, self.nvx, self.nx))
            for k in range(0, self.nx):
                Work = (self.Internal.fHp_hat[:,:,k]*nHP[k]).reshape((self.nvr*self.nvx), order='F')
                self.Internal.Alpha_CX[:,:,k] = self.Internal.SIG_CX @ Work
            if self.Do_Alpha_CX_Test:
                alpha_cx_test = sigmav_cx_hh(THP_mu, self.Internal.EH2_P) / self.vth
                for k in range(0, self.nx):
                    alpha_cx_test[:,:,k] = alpha_cx_test[:,:,k]*nHP[k]
                    print('Compare alpha_cx and alpha_cx_test')
                    input()

        return
    

    def _compute_alpha_h2_p(self, nHP):
        '''
        Compute H2:P Elastic momentum transfer frequency using Eq.(2.12)
        '''
        if self.debrief > 1:
            print(self.prompt, 'Computing Alpha_H2_P')
        self.Internal.Alpha_H2_P = np.zeros((self.nvr, self.nvx, self.nx))
        ni = self.mesh.ne
        if self.ni_correct:
            ni = np.maximum((self.mesh.ne-nHP), 0)
        # print("ni", ni)
        for k in range(0, self.nx):
            Work = (self.Internal.fi_hat[:,:,k]*ni[k]).reshape((self.nvr*self.nvx), order='F')
            self.Internal.Alpha_H2_P[:,:,k] = (self.Internal.SIG_H2_P @ Work).reshape(self.Internal.Alpha_H2_P[:,:,k].shape, order='F')

        return


    def _compute_alpha_h_h2(self, fH):
        '''
        Compute H2:H Elastic momentum transfer frequency using Eq.(2.13)
        '''
        if self.debrief > 1:
            print(self.prompt, 'Computing Alpha_H2_H')
        
        # Compute Alpha_H2_H for inputed fH, if it is needed and has not
        # already been computed with the present input parameters

        self.Internal.Alpha_H2_H = np.zeros((self.nvr, self.nvx, self.nx))
        for k in range(0, self.nx):
            Work = fH[:,:,k].reshape((self.nvr*self.nvx), order='F')
            self.Internal.Alpha_H2_H[:,:,k] = (self.Internal.SIG_H2_H @ Work).reshape(self.Internal.Alpha_H2_H[:,:,k].shape, order='F')

        return
    



    # ------ Computational Functions ------

    def _compute_omega_values(self, fH2, nH2):
        '''
        Compute elastic momentum transfer frequencies (omega) using Eqs.(2.12-2.14)
        '''
        # Compute Omega values if nH2 is non-zero 

        Omega_H2_P = np.zeros(self.nx)
        Omega_H2_H = np.zeros(self.nx)
        Omega_H2_H2 = np.zeros(self.nx)

        ii = np.argwhere(nH2 <= 0) 
        if ii.size <= 0:
            VxH2 = np.zeros((self.nx))
            # compute VxH2
            if self.COLLISIONS.H2_P_EL or self.COLLISIONS.H2_H_EL or self.COLLISIONS.H2_H2_EL:
                for k in range(0, self.nx):
                    VxH2[k] = self.vth*np.sum(self.dvr_vol*(fH2[:,:,k] @ (self.mesh.vx*self.dvx))) / nH2[k]

            # compute Omega_H2_P for present fH2 and Alpha_H2_P if H2_P elastic collisions are included
            if self.COLLISIONS.H2_P_EL:
                if self.debrief > 1:
                    print(self.prompt, 'Computing Omega_H2_P')
                for k in range(0, self.nx):
                    DeltaVx = (VxH2[k] - self.vxi[k])/self.vth
                    MagDeltaVx = np.maximum(np.abs(DeltaVx), self.DeltaVx_tol)
                    DeltaVx = np.sign(DeltaVx)*MagDeltaVx
                    Omega_H2_P[k] = np.sum(self.dvr_vol*(((self.Internal.Alpha_H2_P[:,:,k]*fH2[:,:,k]) @ self.dvx)))/(nH2[k]*DeltaVx)
                Omega_H2_P =  np.maximum(Omega_H2_P, 0)

            # Compute Omega_H2_H for present fH2 and Alpha_H2_H if H2_H elastic collisions are included
            if self.COLLISIONS.H2_H_EL:
                if self.debrief>1:
                    print(self.prompt+'Computing Omega_H2_H')
                for k in range(self.nx):
                    DeltaVx = (VxH2[k] - self.H_Moments.VxH[k])/self.vth
                    MagDeltaVx = np.maximum(np.abs(DeltaVx), self.DeltaVx_tol)
                    DeltaVx = np.sign(DeltaVx)*MagDeltaVx
                    Omega_H2_H[k] = np.sum(self.dvr_vol*((self.Internal.Alpha_H2_H[:,:,k]*fH2[:,:,k]) @ self.dvx)/(nH2[k]*DeltaVx))
                Omega_H2_H = np.maximum(Omega_H2_H, 0)

            # Compute Omega_H2_H2 for present fH2 if H2_H2 elastic collisions are included
            if self.COLLISIONS.H2_H2_EL:
                if self.debrief > 1:
                    print(self.prompt, 'Computing Omega_H2_H2')

                Wperp_paraH2 = np.zeros(self.nx)
                vr2_2vx_ran2 = np.zeros((self.nvr,self.nvx))
                if np.sum(self.Internal.MH2_H2_sum) < 0:
                    for k in range(0, self.nx):
                        for i in range(0, self.nvr):
                            vr2_2vx_ran2[i,:] = self.mesh.vr[i]**2 - 2*(self.mesh.vx - VxH2[k]/self.vth)**2
                        Wperp_paraH2[k] = np.sum(self.dvr_vol*((vr2_2vx_ran2*fH2[:,:,k]) @ self.dvx))/nH2[k]
                else:
                    for k in range(0, self.nx):
                        M_fH2 = self.Internal.MH2_H2_sum[:,:,k] - fH2[:,:,k]
                        Wperp_paraH2[k] = -np.sum(self.dvr_vol*((self.vr2_2vx2_2D*M_fH2) @ self.dvx))/nH2[k]

                for k in range(0, self.nx):
                    Work = fH2[:,:,k].reshape((self.nvr*self.nvx), order='F')
                    Alpha_H2_H2 = (self.Internal.SIG_H2_H2 @ Work).reshape((self.nvr,self.nvx), order='F')
                    Wpp = Wperp_paraH2[k]
                    MagWpp = np.maximum(abs(Wpp), self.Wpp_tol)
                    Wpp = np.sign(Wpp)*MagWpp  
                    Omega_H2_H2[k] = np.sum(self.dvr_vol*((Alpha_H2_H2*Work.reshape((self.nvr,self.nvx), order='F')) @ self.dvx))/(nH2[k]*Wpp)
                    
                Omega_H2_H2 = np.maximum(Omega_H2_H2, 0)

        return CollisionType(Omega_H2_H2, Omega_H2_P, Omega_H2_H)
    

    def _compute_collision_frequency(self, collision_freqs, gamma_wall):
        '''
        Computes total elastic scattering frequency (Eq. 2.15) 
        and total collision frequency (Eq. 2.16) 
        '''
        # Total Elastic scattering frequency
        Omega_EL = collision_freqs.H2_P + collision_freqs.H2_H + collision_freqs.H2_H2

        # Total collision frequency
        alpha_c = np.zeros((self.nvr,self.nvx,self.nx))
        if self.COLLISIONS.H2_P_CX:
            for k in range(0, self.nx):
                alpha_c[:,:,k] = self.Internal.Alpha_CX[:,:,k] + self.Internal.Alpha_Loss[k] + Omega_EL[k] + gamma_wall[:,:,k]
        else: 
            for k in range(0, self.nx):
                alpha_c[:,:,k] = self.Internal.Alpha_Loss[k]+Omega_EL[k]+gamma_wall[:,:,k]

        self._test_grid_spacing(alpha_c)

        return alpha_c


    def _compute_mesh_equation_coefficients(self, alpha_c, SH2):
        '''
        Define parameters Ak, Bk, Ck, Dk, Fk, Gk using Eqs. (2.22), (2.25), (2.30), (2.33) 
        '''
        Ak = np.zeros((self.nvr,self.nvx,self.nx))
        Bk = np.zeros((self.nvr,self.nvx,self.nx))
        Ck = np.zeros((self.nvr,self.nvx,self.nx))
        Dk = np.zeros((self.nvr,self.nvx,self.nx))
        Fk = np.zeros((self.nvr,self.nvx,self.nx))
        Gk = np.zeros((self.nvr,self.nvx,self.nx))

        for k in range(0, self.nx-1):
            for j in self.vx_pos:
                denom = 2*self.mesh.vx[j] + (self.mesh.x[k+1] - self.mesh.x[k])*alpha_c[:,j,k+1]
                Ak[:,j,k] = (2*self.mesh.vx[j] - (self.mesh.x[k+1] - self.mesh.x[k])*alpha_c[:,j,k]) / denom
                Bk[:,j,k] = (self.mesh.x[k+1] - self.mesh.x[k]) / denom
                Fk[:,j,k] = (self.mesh.x[k+1] - self.mesh.x[k])*self.Internal.fw_hat[:,j]*(SH2[k+1]+SH2[k]) / (self.vth*denom)
        for k in range(1, self.nx):
            for j in self.vx_neg:
                denom = -2*self.mesh.vx[j] + (self.mesh.x[k] - self.mesh.x[k-1])*alpha_c[:,j,k-1]
                Ck[:,j,k] = (-2*self.mesh.vx[j] - (self.mesh.x[k] - self.mesh.x[k -1])*alpha_c[:,j,k]) / denom
                Dk[:,j,k] = (self.mesh.x[k] - self.mesh.x[k-1]) / denom
                Gk[:,j,k] = (self.mesh.x[k] - self.mesh.x[k-1])*self.Internal.fw_hat[:,j]*(SH2[k]+SH2[k-1]) / (self.vth*denom)
        
        return MeshEqCoefficients(Ak, Bk, Ck, Dk, Fk, Gk)
    

    def _compute_swall(self, fH2G, gamma_wall):
        '''
        Compute swall using Eq. (2.15c)
        '''
        Swall = np.zeros((self.nvr, self.nvx, self.nx))
        if np.sum(gamma_wall) > 0:
            if self.debrief > 1:
                print(self.prompt, 'Computing Swall')
            for k in range(0, self.nx): 
                Swall[:,:k] = self.Internal.fw_hat*np.sum(self.dvr_vol*((gamma_wall[:,:,k]*fH2G[:,:,k]) @ self.dvx))
        
        return Swall
    

    def _compute_beta_cx(self, fH2G, nHP):
        '''
        Compute charge exchange source (beta_cx) with Eq. (2.11a) or (2.11b)
        '''
        Beta_CX = np.zeros((self.nvr,self.nvx,self.nx))
        if self.COLLISIONS.H2_P_CX: 
            if self.debrief > 1:
                print(self.prompt, 'Computing Beta_CX')

            if self.COLLISIONS.Simple_CX:
                # Option (B): Compute charge exchange source with assumption that CX source neutrals have  molecular ion distribution function
                # Eq.(2.11b)
                for k in range(0, self.nx): 
                    Beta_CX[:,:,k] = self.Internal.fHp_hat[:,:,k]*np.sum(self.dvr_vol*((self.Internal.Alpha_CX[:,:,k]*fH2G[:,:,k]) @ self.dvx))
            else: 
                # Option (A): Compute charge exchange source using fH2 and vr x sigma x v_v at each velocity mesh point
                # Eq.(2.11a)
                for k in range(0, self.nx):
                    Work = fH2G[:,:,k]
                    Beta_CX[:,:,k] = nHP[k]*self.Internal.fHp_hat[:,:,k]*(self.Internal.SIG_CX @ Work)

        return Beta_CX
    
    def _compute_mh_values(self, fH2G, nH):
        '''
        Compute collision distributions using Eqs. (2.6)-(2.8)
        '''
        MH2_H2 = np.zeros((self.nvr,self.nvx,self.nx))
        MH2_P = np.zeros((self.nvr,self.nvx,self.nx))
        MH2_H = np.zeros((self.nvr,self.nvx,self.nx))
        VxH2G = np.zeros(self.nx)
        TH2G = np.zeros(self.nx)
        if self.COLLISIONS.H2_H2_EL or self.COLLISIONS.H2_P_EL or self.COLLISIONS.H2_H_EL:
            # Compute VxH2G, TH2G

            vr2vx2_ran2 = np.zeros((self.nvr,self.nvx))
            for k in range(0, self.nx):
                VxH2G[k] = self.vth*np.sum(self.dvr_vol*(fH2G[:,:,k] @ (self.mesh.vx * self.dvx))) / nH[k]
                vr2vx2_ran2 = self.mesh.vr[:, None]**2 + (self.mesh.vx[None, :] - VxH2G[k]/self.vth)**2
                TH2G[k] = (2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((vr2vx2_ran2*fH2G[:,:,k]) @ self.dvx)) / (3*CONST.Q*nH[k])

            if self.COLLISIONS.H2_H2_EL:
                if self.debrief > 1: 
                    print(self.prompt, 'Computing MH2_H2')
                # Compute MH2_H2
                Maxwell = create_shifted_maxwellian(self.mesh.vr, self.mesh.vx, TH2G, VxH2G, self.mu, 2, self.mesh.Tnorm)
                MH2_H2 = Maxwell*nH

            if self.COLLISIONS.H2_P_EL:
                if self.debrief > 1:
                    print(self.prompt, 'Computing MH2_P')
                # Compute MH2_P
                vx_shift = (2*VxH2G + self.vxi) / 3
                Tmaxwell = TH2G + (4/9)*(self.mesh.Ti - TH2G + ((self.mu*CONST.H_MASS*(self.vxi - VxH2G)**2) / (6*CONST.Q)))
                Maxwell = create_shifted_maxwellian(self.mesh.vr, self.mesh.vx, Tmaxwell, vx_shift, self.mu, 2, self.mesh.Tnorm)
                MH2_P = Maxwell*nH

            if self.COLLISIONS.H2_H_EL:
                if self.debrief > 1:
                    print(self.prompt, 'Computing MH2_H')
                #Compute MH2_H
                vx_shift = (2*VxH2G + self.H_Moments.VxH) / 3
                Tmaxwell = TH2G + (4/9)*(self.H_Moments.TH - TH2G + ((self.mu*CONST.H_MASS*(self.H_Moments.VxH - VxH2G)**2) / (6*CONST.Q)))
                Maxwell = create_shifted_maxwellian(self.mesh.vr, self.mesh.vx, Tmaxwell, vx_shift, self.mu, 2, self.mesh.Tnorm)
                MH2_H = Maxwell*nH

        return CollisionType(MH2_H2, MH2_P, MH2_H)
    



    # ------ Error Computation ------

    def _compute_final_errors(self, results, SH2, Beta_CX_sum, Swall_sum, m_sums, alpha_c, collision_freqs):

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

        if self.debrief > 1:
            print(self.prompt, 'Computing Collision Operator, Mesh, and Moment Normalized Errors')

        Sloss2 = self.vth*self.Internal.Alpha_Loss*results.nH2 
        for k in range(0, self.nx):
            self.Errors.C_Error[k] = np.abs(results.Sloss[k] - Sloss2[k])/np.max(np.abs(np.array([results.Sloss[k], Sloss2[k]])))

        # Test conservation of particles for charge exchange operator
        if self.COLLISIONS.H2_P_CX:
            for k in range(0, self.nx):
                CX_A = np.sum(self.dvr_vol*((self.Internal.Alpha_CX[:,:,k]*results.fH2[:,:,k]) @ self.dvx))
                CX_B = np.sum(self.dvr_vol*(Beta_CX_sum[:,:,k] @ self.dvx))
                self.Errors.CX_Error[k] = np.abs(CX_A - CX_B)/np.max(np.abs(np.array([CX_A, CX_B])))

        # Test conservation of particles for wall collision operator
        if np.sum(self.mesh.PipeDia) > 0: #NOTE Not Tested Yet
            for k in range(0, self.nx):
                Wall_A = results.WallH2[k]
                Wall_B = np.sum(self.dvr_vol*(Swall_sum[:,:,k] @ self.dvx))
                if np.max(np.abs(np.array([Wall_A, Wall_B]))) > 0:
                    Wall_error[k] = np.abs(Wall_A - Wall_B)/np.max(np.abs(np.array([Wall_A, Wall_B])))

        # Test conservation of particles, x momentum, and total energy of elastic collision operators
        for m in range(0, 3):
            for k in range(0, self.nx):
                if m < 2:
                    TfH2 = np.sum(self.dvr_vol*(results.fH2[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                else:
                    TfH2 = np.sum(self.dvr_vol*((self.Internal.vr2vx2[:,:,k]*results.fH2[:,:,k]) @ self.dvx))

                if self.COLLISIONS.H2_H2_EL:
                    if m < 2:
                        TH2_H2 = np.sum(self.dvr_vol*(self.Internal.MH2_H2_sum[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                    else:
                        TH2_H2 = np.sum(self.dvr_vol*((self.Internal.vr2vx2[:,:,k]*self.Internal.MH2_H2_sum[:,:,k]) @ self.dvx))
                    self.Errors.H2_H2_error[k,m] = np.abs(TfH2 - TH2_H2)/np.max(np.abs(np.array([TfH2, TH2_H2])))
                
                if self.COLLISIONS.H2_H_EL:
                    if m < 2:
                        TH2_H = np.sum(self.dvr_vol*(m_sums.H2_H[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                    else:
                        TH2_H = np.sum(self.dvr_vol*((self.Internal.vr2vx2[:,:,k]*m_sums.H2_H[:,:,k]) @ self.dvx))
                    H2_H_error[k,m] = np.abs(TfH2 - TH2_H)/np.max(np.abs(np.array([TfH2, TH2_H])))

                if self.COLLISIONS.H2_P_EL:
                    if m < 2:
                        TH2_P = np.sum(self.dvr_vol*(m_sums.H2_P[:,:,k] @ (self.dvx*(self.mesh.vx**m))))
                    else:
                        TH2_P = np.sum(self.dvr_vol*((self.Internal.vr2vx2[:,:,k]*m_sums.H2_P[:,:,k]) @ self.dvx))
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
                _Sig[:] = (self.Internal.v_v*sigma_cx_hh(self.Internal.v_v2*(CONST.H_MASS*(self.vth**2)/CONST.Q))).reshape(_Sig.shape, order='F')
                SIG_VX_CX = np.zeros((self.nvr*self.nvx,self.nvr*self.nvx))
                SIG_VX_CX[:] = (self.Internal.Vr2pidVrdVx*self.Internal.vx_vx*((_Sig @ self.dtheta).reshape(self.Internal.vx_vx.shape, order='F'))).reshape(SIG_VX_CX.shape, order='F')
                alpha_vx_cx = np.zeros((self.nvr,self.nvx,self.nx))

                for k in range(0, self.nx):
                    Work = (results.nHP[k]*self.Internal.fHp_hat[:,:,k]).reshape((self.nvr*self.nvx), order='F')
                    alpha_vx_cx[:,:,k] = (SIG_VX_CX @ Work).reshape(alpha_vx_cx[:,:,k].shape, order='F')

                RxCI_CX = np.zeros(self.nx)
                for k in range(0, self.nx):
                    RxCI_CX[k] = -(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((alpha_vx_cx[:,:,k]*results.fH2[:,:,k]) @ self.dvx))

                norm = np.max(np.abs(np.array([self.Output.RxH2CX, RxCI_CX])))
                CI_CX_error = np.zeros(self.nx)
                for k in range(0, self.nx):
                    CI_CX_error[k] = np.abs(self.Output.RxH2CX[k] - RxCI_CX[k])/norm

                print(self.prompt,'Maximum normalized momentum transfer error in CX collision operator: ', sval(np.max(CI_CX_error)))

            if self.COLLISIONS.H2_P_EL: # P -> H2 momentum transfer via full collision integral
                RxCI_P_H2 = np.zeros(self.nx)
                for k in range(0, self.nx):
                    RxCI_P_H2[k] = -(1/3)*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((self.Internal.Alpha_H2_P[:,:,k] * results.fH2[:,:,k]) @ self.dvx))

                norm = np.max(np.abs(np.array([self.Output.RxP_H2, RxCI_P_H2])))
                CI_P_H2_error = np.zeros(self.nx)
                for k in range(0, self.nx):
                    CI_P_H2_error[k] = np.abs(self.Output.RxP_H2[k] - RxCI_P_H2[k])/norm 

                print(self.prompt, 'Maximum normalized momentum transfer error in P -> H2 elastic BKG collision operator: ', sval(np.max(CI_P_H2_error)))
            
            if self.COLLISIONS.H2_H_EL: # H -> H2 momentum transfer via full collision integral
                RxCI_H_H2 = np.zeros(self.nx)
                for k in range(0, self.nx):
                    RxCI_H_H2[k] = -(1/3)*(2*self.mu*CONST.H_MASS)*(self.vth**2)*np.sum(self.dvr_vol*((self.Internal.Alpha_H2_H[:,:,k]*results.fH2[:,:,k]) @ self.dvx))
                
                norm = np.max(np.abs(np.array([self.Output.RxH_H2, RxCI_H_H2])))
                CI_H_H2_error = np.zeros(self.nx)
                for k in range(0, self.nx):
                    CI_H_H2_error[k] = np.abs(self.Output.RxH_H2[k] - RxCI_H_H2[k])/norm
                
                print(self.prompt, 'Maximum normalized momentum transfer error in H -> H2 elastic BKG collision operator: ', sval(np.max(CI_H_H2_error)))
            
            if self.COLLISIONS.H2_H2_EL: # H2 -> H2 perp/parallel energy transfer via full collision integral
                Epara_Perp_CI = np.zeros(self.nx)
                for k in range(0, self.nx):
                    Work = results.fH2[:,:,k].reshape((self.nvr*self.nvx), order='F')
                    Alpha_H2_H2 = (self.Internal.SIG_H2_H2 @ Work).reshape((self.nvr,self.nvx), order='F')
                    Epara_Perp_CI[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**3)*np.sum(self.dvr_vol*((Alpha_H2_H2*results.fH2[:,:,k]) @ self.dvx)) 
                
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
                T1[:,j,k] = 2*self.mesh.vx[j]*(results.fH2[:,j,k+1] - results.fH2[:,j,k])/(self.mesh.x[k+1] - self.mesh.x[k]) 
            T2[:,:,k] = self.Internal.fw_hat[:,:]*(SH2[k+1] + SH2[k])/self.vth
            T3[:,:,k] = Beta_CX_sum[:,:,k+1] + Beta_CX_sum[:,:,k]
            T4[:,:,k] = alpha_c[:,:,k+1]*results.fH2[:,:,k+1] + alpha_c[:,:,k]*results.fH2[:,:,k]
            T5[:,:,k] = collision_freqs.H2_P[k+1]*m_sums.H2_P[:,:,k+1] + collision_freqs.H2_H[k+1]*m_sums.H2_H[:,:,k+1] + collision_freqs.H2_H2[k+1]*self.Internal.MH2_H2_sum[:,:,k+1] + \
                    collision_freqs.H2_P[k]*m_sums.H2_P[:,:,k] + collision_freqs.H2_H[k]*m_sums.H2_H[:,:,k] + collision_freqs.H2_H2[k]*self.Internal.MH2_H2_sum[:,:,k]
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
            qxH2_total2[k] = 0.5*(2*self.mu*CONST.H_MASS)*(self.vth**3)*np.sum(self.dvr_vol*((self.Internal.vr2vx2[:,:,k]*results.fH2[:,:,k]) @ (self.mesh.vx*self.dvx)))
        self.Errors.qxH2_total_error = np.abs(results.qxH2_total - qxH2_total2) / np.max(np.abs(np.array([results.qxH2_total, qxH2_total2])))

        # Compute error in QH2_total
        Q1 = np.zeros(self.nx)
        Q2 = np.zeros(self.nx)
        self.Errors.QH2_total_error = np.zeros(self.nx)
        for k in range(0, self.nx-1):
            Q1[k] = (results.qxH2_total[k+1] - results.qxH2_total[k]) / (self.mesh.x[k+1] - self.mesh.x[k])
            Q2[k] = 0.5*(results.QH2_total[k+1] + results.QH2_total[k])
        self.Errors.QH2_total_error = np.abs(Q1 - Q2)/np.max(np.abs(np.array([Q1, Q2])))

        if self.debrief > 0:
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
            if self.debug > 0:
                input()



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
        
        return
    

    def _test_grid_spacing(self, alpha_c):

        # Test x grid spacing based on Eq.(2.27) in notes
        if self.debrief > 1: 
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
        
        return