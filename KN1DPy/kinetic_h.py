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
from .common.Kinetic_H import Kinetic_H_Common

def kinetic_h(mesh: KineticMesh, mu: int, vxi: NDArray, fHBC: NDArray, GammaxHBC: float, fH2: NDArray = None, fSH: NDArray = None, 
              fH: NDArray = None, nHP: NDArray = None, THP: NDArray = None, jh_coeffs : JH_Coef = None, KH : Kinetic_H_Common = None, 
              truncate: float = 1e-4, max_gen = 50, ni_correct = 0, compute_errors = 0, no_recomb = 0, plot = 0, debug = 0, debrief = 0, pause = 0):
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
        no_recomb : bool, default=false
            If true, does not include recombination as a source of atomic neutrals in the algorithm
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
    fH,nH,GammaxH,VxH,pH,TH,qxH,qxH_total,NetHSource,Sion,QH,RxH,QH_total,AlbedoH,WallH,error

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
        error : int
            Error status
            - 0=no error, solution returned
            - 1=error, no solution returned
            NOTE Change to bool, or replace with raises

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

    #NOTE Temporarily store in old variable, replace later
    vx = mesh.vx
    vr = mesh.vr
    x = mesh.x
    Tnorm = mesh.Tnorm
    Ti = mesh.Ti
    Te = mesh.Te
    n = mesh.ne
    PipeDia = mesh.PipeDia

    COLLISIONS = get_config()['collisions']
    H_H_EL = COLLISIONS['H_H_EL']
    H_P_EL = COLLISIONS['H_P_EL']
    _H_H2_EL = COLLISIONS['H2_H_EL']
    H_P_CX = COLLISIONS['H_P_CX']
    Simple_CX = COLLISIONS['SIMPLE_CX']

    ion_rate_option = get_config()['kinetic_h']['ion_rate']
    if ion_rate_option == 'collrad':
        No_Johnson_Hinnov = False
        Use_Collrad_Ionization = True
    elif ion_rate_option == 'jh':
        No_Johnson_Hinnov = False
        Use_Collrad_Ionization = False
    else:
        No_Johnson_Hinnov = True
        Use_Collrad_Ionization = False


    prompt = 'Kinetic_H => '

    #	Set Kinetic_H_input common block variables 
    vx_s = KH.Input.vx_s
    vr_s = KH.Input.vr_s
    x_s = KH.Input.x_s
    Tnorm_s = KH.Input.Tnorm_s
    mu_s = KH.Input.mu_s
    Ti_s = KH.Input.Ti_s
    Te_s = KH.Input.Te_s
    n_s = KH.Input.n_s
    vxi_s = KH.Input.vxi_s
    fHBC_s = KH.Input.fHBC_s
    GammaxHBC_s = KH.Input.GammaxHBC_s
    PipeDia_s = KH.Input.PipeDia_s
    fH2_s = KH.Input.fH2_s
    fSH_s = KH.Input.fSH_s
    nHP_s = KH.Input.nHP_s
    THP_s = KH.Input.THP_s
    fH_s = KH.Input.fH_s
    Simple_CX_s = KH.Input.Simple_CX_s
    JH_s = KH.Input.JH_s
    Collrad_s = KH.Input.Collrad_s
    Recomb_s = KH.Input.Recomb_s
    H_H_EL_s = KH.Input.H_H_EL_s
    H_P_EL_s = KH.Input.H_P_EL_s
    H_H2_EL_s = KH.Input.H_H2_EL_s
    H_P_CX_s = KH.Input.H_P_CX_s

    #	Kinetic_H_internal common block
    vr2vx2 = KH.Internal.vr2vx2
    vr2vx_vxi2 = KH.Internal.vr2vx_vxi2
    fi_hat = KH.Internal.fi_hat
    ErelH_P = KH.Internal.ErelH_P
    Ti_mu = KH.Internal.Ti_mu
    ni = KH.Internal.ni
    sigv = KH.Internal.sigv
    alpha_ion = KH.Internal.alpha_ion
    v_v2 = KH.Internal.v_v2
    v_v = KH.Internal.v_v
    vr2_vx2 = KH.Internal.vr2_vx2
    vx_vx = KH.Internal.vx_vx
    Vr2pidVrdVx = KH.Internal.Vr2pidVrdVx
    SIG_CX = KH.Internal.SIG_CX
    SIG_H_H = KH.Internal.SIG_H_H
    SIG_H_H2 = KH.Internal.SIG_H_H2
    SIG_H_P = KH.Internal.SIG_H_P
    Alpha_CX = KH.Internal.Alpha_CX
    Alpha_H_H2 = KH.Internal.Alpha_H_H2
    Alpha_H_P = KH.Internal.Alpha_H_P
    MH_H_sum = KH.Internal.MH_H_sum
    Delta_nHs = KH.Internal.Delta_nHs
    Sn = KH.Internal.Sn
    Rec = KH.Internal.Rec

    #	Kinetic_H_Moments
    
    nH2 = KH.Moments.nH2
    vxH2 = KH.Moments.VxH2
    TH2 = KH.Moments.TH2

    #	Internal Debug switches

    Shifted_Maxwellian_Debug = 0
    CI_Test = 1
    Do_Alpha_CX_Test = 0

    #	Internal Tolerances

    DeltaVx_tol = .01
    Wpp_tol = .001

    #	Test input parameters

    if debug>0:
        #plot = np.max(plot, 1)
        debrief = np.maximum(debrief, 1)
        pause = 1
    JH = 1
    if No_Johnson_Hinnov:
        JH = 0
    Recomb = 1
    if no_recomb:
        Recomb = 0
    error = 0

    nvr = vr.size
    nvx = vx.size
    nx = x.size

    dx = x-np.roll(x,1)
    dx = dx[1:]
    notpos = np.argwhere(dx <= 0)
    if notpos.size > 0:
        raise Exception(prompt+'x[*] must be increasing with index!')
    if nvx % 2 != 0:
        raise Exception(prompt+'Number of elements in vx must be even!') 
    if Ti.size != nx:
        raise Exception(prompt+'Number of elements in Ti and x do not agree!')
    if vxi is None:
        vxi = np.zeros(nx)
    if vxi.size != nx:
        raise Exception(prompt+'Number of elements in vxi and x do not agree!')
    if Te.size != nx:
        raise Exception(prompt+'Number of elements in Te and x do not agree!')
    if n.size != nx:
        raise Exception('Number of elements in n and x do not agree!')
    if GammaxHBC is None:
        raise Exception(prompt+'GammaxHBC is not defined!')
    if PipeDia is None:
        PipeDia = np.zeros(nx)
    if PipeDia.size != nx:
        raise Exception('Number of elements in PipeDia and x do not agree!')
    if len(fHBC[:,0]) != nvr:
        raise Exception(prompt+'Number of elements in fHBC[:,0] and vr do not agree!')
    if len(fHBC[0,:]) != nvx:
        raise Exception(prompt+'Number of elements in fHBC[0,:] and vx do not agree!')
    if fH2 is None:
        fH2 = np.zeros((nvr, nvx, nx))
    if fH2[:,0,0].size != nvr:
        raise Exception(prompt+'Number of elements in fH2[:,0,0] and vr do not agree!')
    if fH2[0,:,0].size != nvx:
        raise Exception(prompt+'Number of elements in fH2[0,:,0] and vx do not agree!')
    if fH2[0,0,:].size != nx:
        raise Exception(prompt+'Number of elements in fH2[0,0,:] and x do not agree!')
    if fSH is None:
        fSH = np.zeros((nvr, nvx, nx))
    if fSH[:,0,0].size != nvr:
        raise Exception(prompt+'Number of elements in fSH[:,0,0] and vr do not agree!')
    if fSH[0,:,0].size != nvx:
        raise Exception(prompt+'Number of elements in fSH[0,:,0] and vx do not agree!')
    if fSH[0,0,:].size != nx:
        raise Exception(prompt+'Number of elements in fSH[0,0,:] and x do not agree!')
    if nHP is None:
        nHP = np.zeros(nx)
    if nHP.size != nx:
        raise Exception(prompt+'Number of elements in nHP and x do not agree!')
    if THP is None:
        THP = np.full(nx, 1.0)
    if THP.size != nx:
        raise Exception(prompt+'Number of elements in nHP and x do not agree!')
    if fH is None:
        fH = np.zeros((nvr,nvx,nx))
    if fH[:,0,0].size != nvr:
        raise Exception(prompt+'Number of elements in fH[:,0,0] and vr do not agree!')
    if fH[0,:,0].size != nvx:
        raise Exception(prompt+'Number of elements in fH[0,:,0] and vx do not agree!')
    if fH[0,0,:].size != nx:
        raise Exception(prompt+'Number of elements in fH[0,0,:] and x do not agree!')
    if np.sum(abs(vr)) <= 0:
        raise Exception(prompt+'vr is all 0!')
    ii = np.argwhere(vr <= 0)
    if ii.size > 0:
        raise Exception(prompt+'vr contains zero or negative element(s)!')
    if np.sum(abs(vx)) <= 0:
        raise Exception(prompt+'vx is all 0!')
    if np.sum(x) <= 0:
        raise Exception(prompt+'Total(x) is less than or equal to 0!')
    if mu is None:
        raise Exception(prompt+'mu is not defined!')
    if mu not in [1,2]:
        raise Exception(prompt+'mu must be 1 or 2!')

    #NOTE Removed Plotting formatting, bring back once the program actually works

    i_n = np.argwhere(vx < 0)
    if np.size(i_n) < 1:
        print(prompt+'vx contains no negative elements!')
    i_p = np.argwhere(vx > 0)
    if np.size(i_p) < 1:
        print(prompt+'vx contains no positive elements!')
    i_z = np.argwhere(vx == 0)
    if np.size(i_z) > 0:
        print(prompt+'vx contains one or more zero elements!')
    diff = np.argwhere(vx[i_p] != -np.flipud(vx[i_n]))
    if diff.size > 0:
        raise Exception(prompt + " vx array elements are not symmetric about zero!")


    fHBC_input = np.zeros(fHBC.shape)
    fHBC_input[:,i_p] = fHBC[:,i_p]
    test = np.sum(fHBC_input)
    if test <= 0.0 and abs(GammaxHBC) > 0:
        raise Exception(prompt+'Values for fHBC[:,:] with vx > 0 are all zero!')

    #	Output variables

    nH = np.zeros(nx)
    GammaxH = np.zeros(nx)
    VxH = np.zeros(nx)
    pH = np.zeros(nx)
    TH = np.zeros(nx)
    qxH = np.zeros(nx)
    qxH_total = np.zeros(nx)
    NetHSource = np.zeros(nx)
    WallH = np.zeros(nx)
    Sion = np.zeros(nx)
    QH = np.zeros(nx)
    RxH = np.zeros(nx)
    QH_total = np.zeros(nx)
    KH.Output.piH_xx = np.zeros(nx)
    KH.Output.piH_yy = np.zeros(nx)
    KH.Output.piH_zz = np.zeros(nx)
    KH.Output.RxHCX = np.zeros(nx)
    KH.Output.RxH2_H = np.zeros(nx)
    KH.Output.RxP_H = np.zeros(nx)
    KH.Output.RxW_H = np.zeros(nx)
    KH.Output.EHCX = np.zeros(nx)
    KH.Output.EH2_H = np.zeros(nx)
    KH.Output.EP_H = np.zeros(nx)
    KH.Output.EW_H = np.zeros(nx)
    KH.Output.Epara_PerpH_H = np.zeros(nx)
    AlbedoH = 0.0e0
    KH.Output.SourceH = np.zeros(nx)
    KH.Output.SRecomb = np.zeros(nx)

    #	Internal variables

    Work = np.zeros((nvr*nvx))
    fHG = np.zeros((nvr,nvx,nx))
    NHG = np.zeros((nx,max_gen+1))
    Vth = np.sqrt((2*CONST.Q*Tnorm) / (mu*CONST.H_MASS))
    Vth2 = Vth*Vth
    Vth3 = Vth2*Vth
    fHs = np.zeros(nx)
    nHs = np.zeros(nx)
    Alpha_H_H = np.zeros((nvr,nvx))
    Omega_H_P = np.zeros(nx)
    Omega_H_H2 = np.zeros(nx)
    Omega_H_H = np.zeros(nx)
    VxHG = np.zeros(nx)
    THG = np.zeros(nx)
    Wperp_paraH = np.zeros(nx)
    vr2vx2_ran2 = np.zeros((nvr,nvx))
    vr2_2vx_ran2 = np.zeros((nvr,nvx))
    vr2_2vx2_2D = np.zeros((nvr,nvx))
    RxCI_CX = np.zeros(nx)
    RxCI_H2_H = np.zeros(nx)
    RxCI_P_H = np.zeros(nx)
    Epara_Perp_CI = np.zeros(nx)
    CI_CX_error = np.zeros(nx)
    CI_H2_H_error = np.zeros(nx)
    CI_P_H_error = np.zeros(nx)
    CI_H_H_error = np.zeros(nx)
    Maxwell = np.zeros((nvr,nvx,nx))

    #NOTE Fix later
    differential = VSpace_Differentials(vr, vx)
    Vr2pidVr = differential.dvr_vol
    dVx = differential.dvx

    #	Vr^2-2*Vx^2

    for i in range(nvr):
        vr2_2vx2_2D[i,:] = (vr[i]**2) - 2*(vx**2)

    #	Theta-prime coordinate

    # Theta-prime Coordinate
    ntheta = 5 # use 5 theta mesh points for theta integration
    dTheta = np.ones(ntheta)/ntheta
    theta = np.pi*(np.arange(ntheta)/ntheta + 0.5/ntheta)
    cos_theta = np.cos(theta)

    #	Scale input molecular distribution function to agree with desired flux
    gamma_input = 1.0
    if abs(GammaxHBC) > 0:
        gamma_input = Vth*np.sum(Vr2pidVr*(fHBC_input @ (vx*dVx)))
    ratio = abs(GammaxHBC)/gamma_input
    fHBC_input = fHBC_input*ratio
    if abs(ratio - 1) > 0.01*truncate:
        fHBC = fHBC_input
    fH[:,i_p,0] = fHBC_input[:,i_p]

    #	if fH2 is zero, then turn off elastic H2 <-> H collisions
    H_H2_EL = _H_H2_EL
    if np.sum(fH2) <= 0:
        H_H2_EL = 0

    #	Set iteration scheme
    fH_iterate = 0
    if (H_H_EL != 0) or (H_P_EL != 0) or (H_H2_EL != 0): 
        fH_iterate = 1

    fH_generations = 0
    if (fH_iterate != 0) or (H_P_CX != 0): 
        fH_generations = 1

    #	Set flags to make use of previously computed local parameters 
    New_Grid = 1
    if vx_s is not None:
        test = 0 
        ii = np.argwhere(vx_s != vx).T ; test = test + np.size(ii)
        ii = np.argwhere(vr_s != vr).T ; test = test + np.size(ii)
        ii = np.argwhere(x_s != x).T ; test = test + np.size(ii)
        ii = np.argwhere(Tnorm_s != Tnorm).T ; test = test + np.size(ii)
        ii = np.argwhere(mu_s != mu).T ; test = test + np.size(ii)
        if test <= 0:
            New_Grid = 0
    New_Protons = 1
    if Ti_s is not None:
        test = 0 
        ii = np.argwhere(Ti_s != Ti).T ; test = test + np.size(ii)
        ii = np.argwhere(n_s != n).T ; test = test + np.size(ii)
        ii = np.argwhere(vxi_s != vxi).T ; test = test + np.size(ii)
        if test <= 0:
            New_Protons = 0
    New_Molecular_Ions = 1
    if nHP_s is not None:
        test = 0
        ii = np.argwhere(nHP_s != nHP).T ; test = test + np.size(ii)
        ii = np.argwhere(THP_s != THP).T ; test = test + np.size(ii)
        if test <= 0:
            New_Molecular_Ions = 0
    New_Electrons = 1
    if Te_s is not None:
        test = 0 
        ii = np.argwhere(Te_s != Te).T ; test = test + np.size(ii)
        ii = np.argwhere(n_s != n).T ; test = test + np.size(ii)
        if test <= 0:
            New_Electrons = 0
    New_fH2 = 1
    if fH2_s is not None:
        ii = np.argwhere(fH2_s != fH2)
        if np.size(ii) <= 0:
            New_fH2 = 0
    New_fSH = 1
    if fSH_s is not None:
        ii = np.argwhere(fSH_s != fSH)
        if np.size(ii) <= 0:
            New_fSH = 0
    New_Simple_CX = 1
    if Simple_CX_s is not None:
        ii = np.argwhere(Simple_CX_s != Simple_CX)
        if np.size(ii) <= 0:
            New_Simple_CX = 0
    New_H_Seed = 1
    if fH_s is not None:
        ii = np.argwhere(fH_s != fH)
        if np.size(ii) <= 0:
            New_H_Seed = 0

    Do_sigv = 			New_Grid | New_Electrons
    Do_ni = 			New_Grid | New_Electrons | New_Protons | New_Molecular_Ions
    Do_fH2_moments = 	(New_Grid | New_fH2) & (np.sum(fH2) > 0.0)
    Do_Alpha_CX = 		(New_Grid | (Alpha_CX is None) | Do_ni | New_Simple_CX) and H_P_CX
    Do_SIG_CX = 		(New_Grid | (SIG_CX is None) | New_Simple_CX) & (Simple_CX == 0) & Do_Alpha_CX
    Do_Alpha_H_H2 = 	(New_Grid | (Alpha_H_H2 is None) | New_fH2) & H_H2_EL
    Do_SIG_H_H2 = 		(New_Grid | (SIG_H_H2 is None)) & Do_Alpha_H_H2
    Do_SIG_H_H = 		(New_Grid | (SIG_H_H is None)) & H_H_EL
    Do_Alpha_H_P = 		(New_Grid | (Alpha_H_P is None) | Do_ni) & H_P_EL
    Do_SIG_H_P = 		(New_Grid | (SIG_H_P is None)) & Do_Alpha_H_P
    Do_v_v2 = 			(New_Grid | (v_v2 is None)) & (CI_Test | Do_SIG_CX | Do_SIG_H_H2 | Do_SIG_H_H | Do_SIG_H_P)
    
    if debug > 0:
        print("Kinetic H Settings")
        print("H_H_EL", H_H_EL)
        print("H_P_EL", H_P_EL)
        print("_H_H2_EL", _H_H2_EL)
        print("H_P_CX", H_P_CX)
        print("New_Grid", New_Grid)
        print("New_Protons", New_Protons)
        print("New_Molecular_Ions", New_Molecular_Ions)
        print("New_Electrons", New_Electrons)
        print("New_fH2", New_fH2)
        print("New_fSH", New_fSH)
        print("New_Simple_CX", New_Simple_CX)
        print("New_H_Seed", New_H_Seed)
        print("Do_sigv", Do_sigv)
        print("Do_ni", Do_ni)
        print("Do_fH2_moments", Do_fH2_moments)
        print("Do_Alpha_CX", Do_Alpha_CX)
        print("Do_SIG_CX", Do_SIG_CX)
        print("Simple_CX", Simple_CX)
        print("Do_Alpha_H_H2", Do_Alpha_H_H2)
        print("Do_SIG_H_H2", Do_SIG_H_H2)
        print("Do_SIG_H_H", Do_SIG_H_H)
        print("Do_Alpha_H_P", Do_Alpha_H_P)
        print("Do_SIG_H_P", Do_SIG_H_P)
        print("Do_v_v2", Do_v_v2)
        input()

    nH2 = np.zeros(nx)
    vxH2 = np.zeros(nx)
    TH2 = np.full(nx, 1.0)

    if Do_fH2_moments:
        if debrief > 1:
            print(prompt+'Computing vx and T moments of fH2')

        #	Compute x flow velocity and temperature of molecular species
        for k in range(nx):
            nH2[k] = np.sum(Vr2pidVr*(fH2[:,:,k] @ dVx))
            if nH2[k] > 0:
                vxH2[k] = Vth*np.sum(Vr2pidVr*(fH2[:,:,k] @ (vx*dVx)))/nH2[k]
                for i in range(nvr):
                    vr2vx2_ran2[i,:] = vr[i]**2 + (vx - vxH2[k]/Vth)**2
                TH2[k] = (2*mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*((vr2vx2_ran2*fH2[:,:,k]) @ dVx)) / (3*CONST.Q*nH2[k])

    if New_Grid:
        if debrief > 1:
            print(prompt+'Computing vr2vx2, vr2vx_vxi2, ErelH_P')

        #	Magnitude of total normalized v^2 at each mesh point
        vr2vx2 = np.zeros((nvr,nvx,nx))
        for i in range(nvr):
            for k in range(nx):
                vr2vx2[i,:,k] = vr[i]**2 + vx**2

        #	Magnitude of total normalized (v-vxi)^2 at each mesh point
        vr2vx_vxi2 = np.zeros((nvr,nvx,nx))
        for i in range(nvr):
            for k in range(nx):
                vr2vx_vxi2[i,:,k] = vr[i]**2 + (vx - vxi[k]/Vth)**2

        #	Atomic hydrogen ion energy in local rest frame of plasma at each mesh point
        ErelH_P = (0.5*CONST.H_MASS*vr2vx_vxi2*Vth2) / CONST.Q
        ErelH_P = np.maximum(ErelH_P, 0.1) # sigmav_cx does not handle neutral energies below 0.1 eV
        ErelH_P = np.minimum(ErelH_P, 2e4) # sigmav_cx does not handle neutral energies above 20 keV

    if New_Protons:
        if debrief>1:
            print(prompt+'Computing Ti/mu at each mesh point')

        #	Ti/mu at each mesh point
        Ti_mu = np.zeros((nvr,nvx,nx))
        for k in range(nx):
            Ti_mu[:,:,k] = Ti[k]/mu

        #	Compute Fi_hat
        if debrief>1:
            print(prompt+'Computing fi_Hat')
        vx_shift = vxi
        Tmaxwell = Ti
        mol = 1
        Maxwell = create_shifted_maxwellian(vr,vx,Tmaxwell,vx_shift,mu,mol,Tnorm)
        fi_hat = np.copy(Maxwell)

    if compute_errors:
        if debrief>1:
            print(prompt+'Computing Vbar_Error')

        #	Test: The average speed of a non-shifted maxwellian should be 2*Vth*sqrt(Ti[x]/Tnorm)/sqrt(pi)

        vx_shift = np.zeros(nx)
        Tmaxwell = Ti
        mol = 1
        Maxwell = create_shifted_maxwellian(vr,vx,Tmaxwell,vx_shift,mu,mol,Tnorm)
        
        vbar_test = np.zeros((nvr,nvx,ntheta))
        vbar_error = np.zeros(nx)
        for m in range(ntheta):
            vbar_test[:,:,m] = vr2vx2[:,:,0]
        _vbar_test = np.zeros((nvr*nvx,ntheta))
        _vbar_test[:] = (Vth*np.sqrt(vbar_test)).reshape(_vbar_test.shape, order='F')
        vbar_test = np.zeros((nvr,nvx))
        vbar_test[:] = (_vbar_test @ dTheta).reshape(vbar_test.shape, order='F')
        for k in range(nx):
            vbar = np.sum(Vr2pidVr*((vbar_test*Maxwell[:,:,k]) @ dVx))
            vbar_exact = 2*Vth*np.sqrt(Ti[k]/Tnorm) / np.sqrt(np.pi)
            vbar_error[k] = abs(vbar-vbar_exact) / vbar_exact
        if debrief > 0:
            print(prompt+'Maximum Vbar error = ', sval(max(vbar_error)))

    if Do_ni:
        if debrief>1:
            print(prompt+'Computing ni profile')
        ni = n
        if ni_correct:
            ni = n-nHP
        ni = np.maximum(ni, 0.01*n)

    if Do_sigv:
        if debrief>1:
            print(prompt+'Computing sigv')

        #	Compute sigmav rates for each reaction with option to use rates
        #	from CR model of Johnson-Hinnov

        sigv = np.zeros((nx,3))

        #	Reaction R1:  e + H -> e + H(+) + e   (ionization)
        #NOTE Replace Use_Collrad_Ionization with constant for consistency across program, check with someone who knows what they are doing if this is correct
        if CONST.USE_COLLRAD_IONIZATION:
            sigv[:,1] = collrad_sigmav_ion_h0(n,Te) # from COLLRAD code (DEGAS-2)
        else:
            #NOTE Replace JH with constant for consistency across program, check with someone who knows what they are doing if this is correct
            if CONST.USE_JH:
                print("using in mesh")
                sigv[:,1] = jhs_coef(n, Te, jh_coeffs, no_null=True) # Johnson-Hinnov, limited Te range; fixed JHS_coef capitalization #NOTE Not tested yet
            else:
                sigv[:,1] = sigmav_ion_h0(Te) # from Janev et al., up to 20keV #NOTE Not Tested Yet
                
        #	Reaction R2:  e + H(+) -> H(1s) + hv  (radiative recombination)
        #NOTE Replace JH with constant for consistency across program, check with someone who knows what they are doing if this is correct
        if CONST.USE_JH:
            sigv[:,2] = jhalpha_coef(n, Te, jh_coeffs, no_null=True)
        else:
            sigv[:,2] = sigmav_rec_h1s(Te)

        #	H ionization rate (normalized by vth) = reaction 1
        alpha_ion = (n*sigv[:,1]) / Vth

        #	Recombination rate (normalized by vth) = reaction 2
        Rec = (n*sigv[:,2]) / Vth

    #	Compute Total Atomic Hydrogen Source
    Sn = np.zeros((nvr,nvx,nx))

    #	Add Recombination (optionally) and User-Supplied Hydrogen Source (velocity space distribution)
    for k in range(nx):
        Sn[:,:,k] = fSH[:,:,k]/Vth
        if Recomb:
            Sn[:,:,k] = Sn[:,:,k] + fi_hat[:,:,k]*ni[k]*Rec[k]

    #	Set up arrays for charge exchange and elastic collision computations, if needed

    if Do_v_v2 == 1:
        if debrief > 1:
            print(prompt+'Computing v_v2, v_v, vr2_vx2, and vx_vx')

        #	v_v2=(v-v_prime)^2 at each double velocity space mesh point, including theta angle
        v_v2 = np.zeros((nvr,nvx,nvr,nvx,ntheta))

        #	vr2_vx2=0.125* [ vr2 + vr2_prime - 2*vr*vr_prime*cos(theta) - 2*(vx-vx_prime)^2 ]
        #		at each double velocity space mesh point, including theta angle
        vr2_vx2 = np.zeros((nvr,nvx,nvr,nvx,ntheta))
        for m in range(ntheta):
            for l in range(nvx):
                for k in range(nvr):
                    for i in range(nvr):
                        v_starter = vr[i]**2 + vr[k]**2 - 2*vr[i]*vr[k]*cos_theta[m]
                        v_v2[i,:,k,l,m] = v_starter + (vx[:] - vx[l])**2
                        vr2_vx2[i,:,k,l,m] = v_starter - 2*(vx[:] - vx[l])**2

        #	v_v=|v-v_prime| at each double velocity space mesh point, including theta angle
        v_v = np.sqrt(v_v2)

        #	vx_vx=(vx-vx_prime) at each double velocity space mesh point
        vx_vx = np.zeros((nvr,nvx,nvr,nvx))
        for j in range(nvx):
            for l in range(nvx):
                vx_vx[:,j,:,l] = vx[j]-vx[l]

        #	Set Vr'2pidVr'*dVx' for each double velocity space mesh point

        Vr2pidVrdVx = np.zeros((nvr,nvx,nvr,nvx))
        for k in range(nvr):
            Vr2pidVrdVx[:,:,k,:] = Vr2pidVr[k]
        for l in range(nvx):
            Vr2pidVrdVx[:,:,:,l] = Vr2pidVrdVx[:,:,:,l]*dVx[l]

    if Simple_CX == 0 and Do_SIG_CX == 1: #NOTE Not Tested Yet
        if debrief>1:
            print(prompt+'Computing SIG_CX')

        #	Option (A) was selected: Compute SigmaV_CX from sigma directly.
        #	In preparation, compute SIG_CX for present velocity space grid, if it has not 
        #	already been computed with the present input parameters

        #	Compute sigma_cx * v_v at all possible relative velocities
        _Sig = np.zeros((nvr*nvx*nvr*nvx, ntheta))
        _Sig[:] = (v_v*sigma_cx_h0(v_v2*(0.5*CONST.H_MASS*Vth2/CONST.Q))).reshape(_Sig.shape, order='F')

        #	Set SIG_CX = vr' x Integral{v_v*sigma_cx} 
        #		over theta=0,2pi times differential velocity space element Vr'2pidVr'*dVx'
        SIG_CX = np.zeros((nvr*nvx, nvr*nvx))
        SIG_CX[:] = (Vr2pidVrdVx*((_Sig @ dTheta).reshape(Vr2pidVrdVx.shape, order='F'))).reshape(SIG_CX.shape, order='F')

        #	SIG_CX is now vr' * sigma_cx(v_v) * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])

    if Do_SIG_H_H == 1:
        if debrief>1:
            print(prompt+'Computing SIG_H_H')

        #	Compute SIG_H_H for present velocity space grid, if it is needed and has not already been computed with the present input parameters

        #	Compute sigma_H_H * vr2_vx2 * v_v at all possible relative velocities
        _Sig = np.zeros((nvr*nvx*nvr*nvx,ntheta))
        _Sig[:] = (vr2_vx2*v_v*sigma_el_h_h(v_v2*(0.5*CONST.H_MASS*mu*Vth2/CONST.Q), vis=True) / 8).reshape(_Sig.shape, order='F')

        #	Note: For viscosity, the cross section for D -> D is the same function of center of mass energy as H -> H.

        #	Set SIG_H_H = vr' x Integral{vr2_vx2*v_v*sigma_H_H} over theta=0,2pi times differential velocity space element Vr'2pidVr'*dVx'
        SIG_H_H = np.zeros((nvr*nvx,nvr*nvx))
        SIG_H_H[:] = (Vr2pidVrdVx*(_Sig @ dTheta).reshape(Vr2pidVrdVx.shape, order='F')).reshape(SIG_H_H.shape, order='F')
        #	SIG_H_H is now vr' * sigma_H_H(v_v) * vr2_vx2 * v_v (intergated over theta) for all possible ([vr,vx],[vr',vx'])

    if Do_SIG_H_H2 == 1:
        if debrief > 1:
            print(prompt+'Computing SIG_H_H2')

        #	Compute SIG_H_H2 for present velocity space grid, if it is needed and has not
        #		already been computed with the present input parameters

        #	Compute sigma_H_H2 * v_v at all possible relative velocities

        _Sig = np.zeros((nvr*nvx*nvr*nvx,ntheta))
        _Sig[:] = (v_v*sigma_el_h_hh(v_v2*(0.5*CONST.H_MASS*Vth2/CONST.Q))).reshape(_Sig.shape, order='F')

        #	NOTE: using H energy here for cross-sections tabulated as H->H2

        #	Set SIG_H_H2 = vr' x vx_vx x Integral{v_v*sigma_H_H2} over theta=0,
        #		2pi times differential velocity space element Vr'2pidVr'*dVx'

        SIG_H_H2 = np.zeros((nvr*nvx,nvr*nvx))
        SIG_H_H2[:] = (Vr2pidVrdVx*vx_vx*(_Sig @ dTheta).reshape(Vr2pidVrdVx.shape, order='F')).reshape(SIG_H_H2.shape, order='F')

        #	SIG_H_H2 is now vr' *vx_vx * sigma_H_H2(v_v) * v_v 
        #		(intergated over theta) for all possible ([vr,vx],[vr',vx'])

    if Do_SIG_H_P == 1:
        if debrief>1:
            print(prompt+'Computing SIG_H_P')

        #	Compute SIG_H_P for present velocity space grid, if it is needed and has not 
        #		already been computed with the present input parameters

        #	Compute sigma_H_P * v_v at all possible relative velocities
        _Sig = np.zeros((nvr*nvx*nvr*nvx,ntheta))
        _Sig[:] = (v_v*sigma_el_p_h(v_v2*(0.5*CONST.H_MASS*Vth2/CONST.Q))).reshape(_Sig.shape, order='F') #NOTE Program slows significantly here

        #	Set SIG_H_P = vr' x vx_vx x Integral{v_v*sigma_H_P} over theta=0,
        #		2pi times differential velocity space element Vr'2pidVr'*dVx'

        SIG_H_P = np.zeros((nvr*nvx,nvr*nvx))
        SIG_H_P[:] = (Vr2pidVrdVx*vx_vx*(_Sig @ dTheta).reshape(Vr2pidVrdVx.shape, order='F')).reshape(SIG_H_P.shape, order='F')

        #	SIG_H_P is now vr' *vx_vx * sigma_H_P(v_v) * v_v (intergated over theta) 
        #		for all possible ([vr,vx],[vr',vx'])


    #	Compute Alpha_CX for present Ti and ni, if it is needed and has not
    #		already been computed with the present parameters
    if Do_Alpha_CX == 1:
        if debrief > 1:
            print(prompt+'Computing Alpha_CX')

        if Simple_CX:
            #	Option (B): Use maxwellian weighted <sigma v>

            #	Charge Exchange sink rate

            Alpha_CX = sigmav_cx_h0(Ti_mu, ErelH_P) / Vth
            for k in range(nx):
                Alpha_CX[:,:,k] = Alpha_CX[:,:,k]*ni[k]

        else: #NOTE Not Tested Yet
            #	Option (A): Compute SigmaV_CX from sigma directly via SIG_CX

            Alpha_CX = np.zeros((nvr,nvx,nx))
            for k in range(nx):
                Work[:] = (fi_hat[:,:,k]*ni[k]).reshape(Work.shape, order='F')
                Alpha_CX[:,:,k] = (SIG_CX @ Work).reshape(Alpha_CX[:,:,k].shape, order='F')
            
            if Do_Alpha_CX_Test:
                Alpha_CX_Test = sigmav_cx_h0(Ti_mu, ErelH_P) / Vth
                for k in range(nx):
                    Alpha_CX_Test[:,:,k] = Alpha_CX_Test[:,:,k]*ni[k]
                print('Compare alpha_cx and alpha_cx_test')
            

    #	Compute Alpha_H_H2 for inputted fH, if it is needed and has not
    #		already been computed with the present input parameters

    if Do_Alpha_H_H2 == 1:
        if debrief > 1:
            print(prompt+'Computing Alpha_H_H2')
        Alpha_H_H2 = np.zeros((nvr,nvx,nx))
        for k in range(nx):
            Work[:] = fH2[:,:,k].reshape(Work.shape, order='F')
            Alpha_H_H2[:,:,k] = (SIG_H_H2 @ Work).reshape(Alpha_H_H2[:,:,k].shape, order='F')

    #	Compute Alpha_H_P for present Ti and ni 
    #		if it is needed and has not already been computed with the present parameters
    if Do_Alpha_H_P == 1:
        if debrief > 1:
            print(prompt+'Computing Alpha_H_P')
        Alpha_H_P = np.zeros((nvr,nvx,nx))
        for k in range(nx):
            Work[:] = (fi_hat[:,:,k]*ni[k]).reshape(Work.shape, order='F')
            Alpha_H_P[:,:,k] = (SIG_H_P @ Work).reshape(Alpha_H_P[:,:,k].shape, order='F')

    #	Compute nH
    for k in range(nx):
        nH[k] = np.sum(Vr2pidVr*(fH[:,:,k] @ dVx))
    
    if New_H_Seed:
        MH_H_sum = np.zeros((nvr,nvx,nx))
        Delta_nHs = 1

    #	Compute Side-Wall collision rate
    gamma_wall = np.zeros((nvr,nvx,nx))
    for k in range(nx):
        if PipeDia[k] > 0:
            for j in range(nvx):
                gamma_wall[:,j,k] = 2*vr / PipeDia[k]

    do_fH_Iterate = True

    #	This is the entry point for fH iteration.
    #	Save 'seed' values for comparison later

    while do_fH_Iterate: #NOTE Alpha_CX done before here, but done inside iteration in kh2, does it change per iteration? Is this an error?
        do_fH_Iterate = False
        fHs = copy.copy(fH)
        nHs = copy.copy(nH)

        #	Compute Omega values if nH is non-zero
        ii = np.argwhere(nH <= 0)
        if ii.size <= 0: #NOTE Not Tested Yet, return on iteration

            #	Compute VxH
            if H_P_EL or H_H2_EL or H_H_EL:
                for k in range(nx):
                    VxH[k] = Vth*np.sum(Vr2pidVr*(fH[:,:,k] @ (vx*dVx))) / nH[k]

            #	Compute Omega_H_P for present fH and Alpha_H_P if H_P elastic collisions are included
            if H_P_EL:
                if debrief > 1:
                    print(prompt+'Computing Omega_H_P')
                for k in range(nx):
                    DeltaVx = (VxH[k] - vxi[k]) / Vth
                    MagDeltaVx = np.maximum(abs(DeltaVx), DeltaVx_tol)
                    DeltaVx = np.sign(DeltaVx)*MagDeltaVx
                    Omega_H_P[k] = np.sum(Vr2pidVr*((Alpha_H_P[:,:,k]*fH[:,:,k]) @ dVx)) / (nH[k]*DeltaVx)
                Omega_H_P = np.maximum(Omega_H_P, 0)

            #	Compute Omega_H_H2 for present fH and Alpha_H_H2 if H_H2 elastic collisions are included

            if H_H2_EL:
                if debrief > 1:
                    print(prompt+'Computing Omega_H_H2')
                for k in range(nx):
                    DeltaVx = (VxH[k] - vxH2[k]) / Vth
                    MagDeltaVx = np.maximum(abs(DeltaVx), DeltaVx_tol)
                    DeltaVx = np.sign(DeltaVx)*MagDeltaVx
                    # print("Mag", fH[:,:,k].T)
                    # input()
                    Omega_H_H2[k] = np.sum(Vr2pidVr*((Alpha_H_H2[:,:,k]*fH[:,:,k]) @ dVx)) / (nH[k]*DeltaVx)
                Omega_H_H2 = np.maximum(Omega_H_H2, 0)

            #	Compute Omega_H_H for present fH if H_H elastic collisions are included

            if H_H_EL:
                if debrief > 1:
                    print(prompt+'Computing Omega_H_H')
                if np.sum(MH_H_sum) <= 0:
                    for k in range(nx):
                        for i in range(nvr):
                            vr2_2vx_ran2[i,:] = vr[i]**2 - 2*((vx - (VxH[k]/Vth))**2)
                        Wperp_paraH[k] = np.sum(Vr2pidVr*((vr2_2vx_ran2*fH[:,:,k]) @ dVx)) / nH[k]
                else:
                    for k in range(nx):
                        M_fH = MH_H_sum[:,:,k] - fH[:,:,k]
                        Wperp_paraH[k] = -np.sum(Vr2pidVr*((vr2_2vx2_2D*M_fH) @ dVx)) / nH[k]
                for k in range(nx):
                    Work[:] = fH[:,:,k].reshape(Work.shape, order='F')
                    Alpha_H_H[:] = (SIG_H_H @ Work).reshape(Alpha_H_H.shape, order='F')
                    Wpp = Wperp_paraH[k]
                    MagWpp = np.maximum(np.abs(Wpp), Wpp_tol)
                    Wpp = np.sign(Wpp)*MagWpp
                    Omega_H_H[k] = np.sum(Vr2pidVr*((Alpha_H_H*Work.reshape(Alpha_H_H.shape, order='F')) @ dVx)) / (nH[k]*Wpp)
                Omega_H_H = np.maximum(Omega_H_H, 0)

        #	Total Elastic scattering frequency

        Omega_EL = Omega_H_P + Omega_H_H2 + Omega_H_H

        #	Total collision frequency

        alpha_c = np.zeros((nvr,nvx,nx))
        if H_P_CX:
            for k in range(nx):
                alpha_c[:,:,k] = Alpha_CX[:,:,k] + alpha_ion[k] + Omega_EL[k] + gamma_wall[:,:,k]
        else:
            for k in range(nx):
                alpha_c[:,:,k] = alpha_ion[k] + Omega_EL[k] + gamma_wall[:,:,k]

        #	Test x grid spacing based on Eq.(27) in notes
        if debrief>1:
            print(prompt+'Testing x grid spacing')
        Max_dx = np.full(nx, 1e32)
        for k in range(nx):
            for j in range(i_p[0][0], nvx): # changed ip to i_p
                Max_dx[k] = np.minimum(Max_dx[k], min(2*vx[j] / alpha_c[:,j,k]))

        dx = np.roll(x,-1) - x
        Max_dxL = Max_dx[0:nx-1]
        Max_dxR = Max_dx[1:nx]
        Max_dx = np.minimum(Max_dxL, Max_dxR)
        ilarge = np.argwhere(Max_dx < dx[0:nx-1])

        if ilarge.size > 0:
            print(prompt+'x mesh spacing is too large!') #NOTE Check Formatting
            debug = 1
            out = ""
            jj = 0

            #	Not sure the output is formatted correctly

            print(' \t    x(k+1)-x(k)   Max_dx(k)\t   x(k+1)-x(k)   Max_dx(k)\t   x(k+1)-x(k)   Max_dx(k)\t   x(k+1)-x(k)   Max_dx(k)\t   x(k+1)-x(k)   Max_dx(k)')
            for ii in range(ilarge.size):
                jj += 1
                out += ((str(ilarge[ii])+' \t')[:8]+(str(x[ilarge[ii]+1]-x[ilarge[ii]])+'        ')[:6]+'        '+str(Max_dx[ilarge[ii]])[:4]+'\t')
                if jj>4:
                    print(out)
                    jj = 0
                    out = "\t"
            if jj>0:
                print(out)
            error = 1
            raise Exception("x mesh spacing is too large")

        #	Define parameters Ak, Bk, Ck, Dk, Fk, Gk

        Ak = np.zeros((nvr,nvx,nx))
        Bk = np.zeros((nvr,nvx,nx))
        Ck = np.zeros((nvr,nvx,nx))
        Dk = np.zeros((nvr,nvx,nx))
        Fk = np.zeros((nvr,nvx,nx))
        Gk = np.zeros((nvr,nvx,nx))

        for k in range(0, nx-1):
            for j in range(i_p[0][0], nvx): # double check some of the ranges in for statements I might have some typos
                denom = 2*vx[j] + (x[k+1] - x[k])*alpha_c[:,j,k+1]
                Ak[:,j,k] = (2*vx[j] - (x[k+1] - x[k])*alpha_c[:,j,k]) / denom
                Bk[:,j,k] = (x[k+1] - x[k]) / denom
                Fk[:,j,k] = (x[k+1] - x[k])*(Sn[:,j,k+1]+Sn[:,j,k]) / denom
        for k in range(1, nx):
            for j in range(0, i_p[0][0]):
                denom = -2*vx[j] + (x[k] - x[k-1])*alpha_c[:,j,k-1]
                Ck[:,j,k] = (-2*vx[j] - (x[k] - x[k -1])*alpha_c[:,j,k]) / denom
                Dk[:,j,k] = (x[k] - x[k-1]) / denom
                Gk[:,j,k] = (x[k] - x[k-1])*(Sn[:,j,k]+Sn[:,j,k-1]) / denom
                        
        #	Compute first-flight (0th generation) neutral distribution function
        Beta_CX_sum = np.zeros((nvr,nvx,nx))
        MH_P_sum = np.zeros((nvr,nvx,nx))
        MH_H2_sum = np.zeros((nvr,nvx,nx))
        MH_H_sum = np.zeros((nvr,nvx,nx))
        igen = 0
        if debrief > 0:
            print(prompt+'Computing atomic neutral generation#'+sval(igen))
        fHG[:,i_p,0] = fH[:,i_p,0]
        for k in range(nx-1):
            fHG[:,i_p,k+1] = fHG[:,i_p,k]*Ak[:,i_p,k] + Fk[:,i_p,k]
        for k in range(nx-1,0,-1):
            fHG[:,i_n,k-1] = fHG[:,i_n,k]*Ck[:,i_n,k] + Gk[:,i_n,k]
                
        #	Compute first-flight neutral density profile
        for k in range(nx):
            NHG[k,igen] = np.sum(Vr2pidVr*(fHG[:,:,k] @ dVx))

        # NOTE Add plotting once program is working

        #	Set total atomic neutral distribution function to first flight generation

        fH = copy.copy(fHG)
        nH = NHG[:,0]

# next_generation #########################################################################################################################################################################
        while True:
            if igen+1 > max_gen or fH_generations == 0:
                if debrief > 0:
                    print(prompt+'Completed '+sval(max_gen)+' generations. Returning present solution...')
                break
            igen += 1
            if debrief > 0:
                print(prompt+'Computing atomic neutral generation#'+sval(igen))

            #	Compute Beta_CX from previous generation

            Beta_CX = np.zeros((nvr,nvx,nx))
            if H_P_CX:
                if debrief>1:
                    print(prompt+'Computing Beta_CX')

                if Simple_CX:
                    #	Option (B): Compute charge exchange source with assumption that CX source 
                    #		neutrals have ion distribution function
                    for k in range(nx):
                        Beta_CX[:,:,k] = fi_hat[:,:,k]*np.sum(Vr2pidVr*((Alpha_CX[:,:,k]*fHG[:,:,k]) @ dVx))
                else:
                    #	Option (A): Compute charge exchange source using fH and vr x sigma x v_v at 
                    #		each velocity mesh point
                    for k in range(nx):
                        Work[:] = fHG[:,:,k]
                        Beta_CX[:,:,k] = ni[k]*fi_hat[:,:,k]*(SIG_CX @ Work)

                #	Sum charge exchange source over all generations
                Beta_CX_sum += Beta_CX

            #	Compute MH from previous generation
            MH_H = np.zeros((nvr,nvx,nx))
            MH_P = np.zeros((nvr,nvx,nx))
            MH_H2 = np.zeros((nvr,nvx,nx))
            OmegaM = np.zeros((nvr,nvx,nx))
            if H_H_EL or H_P_EL or H_H2_EL:

                #	Compute VxHG, THG
                for k in range(0, nx):
                    VxHG[k] = Vth*np.sum(Vr2pidVr*(fHG[:,:,k] @ (vx*dVx))) / NHG[k,igen-1]
                    for i in range(0, nvr):
                        vr2vx2_ran2[i,:] = vr[i]**2 + (vx - VxHG[k]/Vth)**2
                    THG[k] = (mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*((vr2vx2_ran2*fHG[:,:,k]) @ dVx)) / (3*CONST.Q*NHG[k,igen-1])

                if H_H_EL:
                    if debrief > 1:
                        print(prompt+'Computing MH_H')

                    #	Compute MH_H 
                    vx_shift = VxHG
                    Tmaxwell = THG
                    mol = 1
                    Maxwell = create_shifted_maxwellian(vr,vx,Tmaxwell,vx_shift,mu,mol,Tnorm)
                    for k in range(nx):
                        MH_H[:,:,k] = Maxwell[:,:,k]*NHG[k,igen-1]
                        OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H_H[k]*MH_H[:,:,k]
                    MH_H_sum += MH_H

                if H_P_EL:
                    if debrief>1:
                        print(prompt+'Computing MH_P')

                    #	Compute MH_P 
                    vx_shift = (VxHG+vxi)/2
                    Tmaxwell = THG + (2/4)*(Ti - THG + mu*CONST.H_MASS*((vxi - VxHG)**2) / (6*CONST.Q))
                    mol = 1
                    Maxwell = create_shifted_maxwellian(vr,vx,Tmaxwell,vx_shift,mu,mol,Tnorm)
                    for k in range(nx):
                        MH_P[:,:,k] = Maxwell[:,:,k]*NHG[k,igen-1]
                        OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H_P[k]*MH_P[:,:,k]
                    MH_P_sum += MH_P

                if H_H2_EL:
                    if debrief>1:
                        print(prompt+'Computing MH_H2')

                    #	Compute MH_H2
                    vx_shift = (VxHG + 2*vxH2)/3
                    Tmaxwell = THG + (4./9.)*(TH2 - THG + 2*mu*CONST.H_MASS*((vxH2 - VxHG)**2) / (6*CONST.Q))
                    mol = 1
                    Maxwell = create_shifted_maxwellian(vr,vx,Tmaxwell,vx_shift,mu,mol,Tnorm)
                    
                    for k in range(nx):
                        MH_H2[:,:,k] = Maxwell[:,:,k]*NHG[k,igen-1]
                        OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H_H2[k]*MH_H2[:,:,k]
                    MH_H2_sum += MH_H2

            #	Compute next generation atomic distribution

            fHG[:] = 0
            for k in range(0, nx-1):
                fHG[:,i_p,k+1] = Ak[:,i_p,k]*fHG[:,i_p,k] + Bk[:,i_p,k]*(Beta_CX[:,i_p,k+1] + OmegaM[:,i_p,k+1] + Beta_CX[:,i_p,k] + OmegaM[:,i_p,k])
            for k in range(nx-1, 0, -1):
                fHG[:,i_n,k-1] = Ck[:,i_n,k]*fHG[:,i_n,k] + Dk[:,i_n,k]*(Beta_CX[:,i_n,k-1] + OmegaM[:,i_n,k-1] + Beta_CX[:,i_n,k] + OmegaM[:,i_n,k])
            for k in range(0, nx):
                NHG[k,igen] = np.sum(Vr2pidVr*(fHG[:,:,k] @ dVx))

            # NOTE Add plotting once program is working

            #	Add result to total neutral distribution function
            fH += fHG
            nH += NHG[:,igen]

            #	Compute 'generation error': Delta_nHG=max(NHG(*,igen)/max(nH))
            #		and decide if another generation should be computed
            Delta_nHG = max(NHG[:,igen]/max(nH))
            if (Delta_nHG < truncate) or (fH_iterate and (Delta_nHG < 0.003*Delta_nHs)):
                #	If fH 'seed' is being iterated, then do another generation until the 'generation error'
                #		is less than 0.003 times the 'seed error' or is less than TRUNCATE
                break

# fH2_done #########################################################################################################################################################################
        
        # NOTE Add plotting once program is working

        #	Compute H density profile
        for k in range(0, nx):
            nH[k] = np.sum(Vr2pidVr*(fH[:,:,k] @ dVx))

        if fH_iterate:

            #	Compute 'seed error': Delta_nHs=(|nHs-nH|)/max(nH) 
            #		If Delta_nHs is greater than 10*truncate then iterate fH
            Delta_nHs = np.max(np.abs(nHs - nH))/np.max(nH)
            if Delta_nHs > 10*truncate:
                do_fH_Iterate = True 

    #	Update Beta_CX_sum using last generation
    if H_P_CX:
        if debrief > 1:
            print(prompt, 'Computing Beta_CX')
        if Simple_CX:
            # Option (B): Compute charge exchange source with assumption that CX source neutrals have
            # ion distribution function
            for k in range(0, nx):
                Beta_CX[:,:,k] = fi_hat[:,:,k]*np.sum(Vr2pidVr*(Alpha_CX[:,:,k]*fHG[:,:,k] @ dVx))
        else:
            # Option (A): Compute charge exchange source using fH2 and vr x sigma x v_v at each velocity mesh point
            for k in range(0, nx):
                Work[:] = fHG[:,:,k]
                Beta_CX[:,:,k] = ni[k]*fi_hat[:,:,k]*(SIG_CX @ Work)
        Beta_CX_sum = Beta_CX_sum + Beta_CX
            
    #	Update MH_*_sum using last generation
    MH_H2 = np.zeros((nvr,nvx,nx))
    MH_P = np.zeros((nvr,nvx,nx))
    MH_H = np.zeros((nvr,nvx,nx))
    OmegaM = np.zeros((nvr,nvx,nx))
    if H_H_EL or H_P_EL or H_H2_EL: 
        # Compute VxH2G, TH2G
        for k in range(0, nx):
            VxHG[k] = Vth*np.sum(Vr2pidVr*(fHG[:,:,k] @ (vx*dVx))) / NHG[k,igen]
            for i in range(0, nvr):
                vr2vx2_ran2[i,:] = vr[i]**2 + (vx - VxHG[k]/Vth)**2
            THG[k] = (mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*((vr2vx2_ran2*fHG[:,:,k]) @ dVx)) / (3*CONST.Q*NHG[k,igen])

        if H_H_EL:
            if debrief > 1: 
                print(prompt, 'Computing MH_H')
            # Compute MH_H
            vx_shift = VxHG
            Tmaxwell = np.copy(THG)
            mol = 1
            Maxwell = create_shifted_maxwellian(vr,vx,Tmaxwell,vx_shift,mu,mol,Tnorm)
            
            for k in range(0, nx):
                MH_H[:,:,k] = Maxwell[:,:,k]*NHG[k,igen]
                OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H_H[k]*MH_H[:,:,k]
            MH_H_sum = MH_H_sum + MH_H

        if H_P_EL:
            if debrief > 1:
                print(prompt, 'Computing MH_P')
            # Compute MH_P
            vx_shift = (VxHG + vxi)/2
            Tmaxwell = THG + (2/4)*(Ti - THG + mu*CONST.H_MASS*((vxi - VxHG)**2) / (6*CONST.Q))
            mol = 1
            Maxwell = create_shifted_maxwellian(vr,vx,Tmaxwell,vx_shift,mu,mol,Tnorm)
            
            for k in range(0, nx):
                MH_P[:,:,k] = Maxwell[:,:,k]*NHG[k,igen]
                OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H_P[k]*MH_P[:,:,k]
            MH_P_sum = MH_P_sum + MH_P

            if H_H2_EL: #NOTE Not Tested Yet
                if debrief > 1:
                    print(prompt, 'Computing MH_H2')
                # Compute MH_H
                vx_shift = (VxHG + 2*vxH2)/3
                Tmaxwell = THG + (4/9)*(TH2 - THG + 2*mu*CONST.H_MASS*((vxH2 - VxHG)**2) / (6*CONST.Q))
                mol = 1
                Maxwell = create_shifted_maxwellian(vr,vx,Tmaxwell,vx_shift,mu,mol,Tnorm)
                for k in range(0, nx):
                    MH_H2[:,:,k] = Maxwell[:,:,k]*NHG[k,igen]
                    OmegaM[:,:,k] = OmegaM[:,:,k] + Omega_H_H2[k]*MH_H2[:,:,k]
                MH_H2_sum = MH_H2_sum + MH_H2

    #	Compute remaining moments

    #NOTE In kinetic_h2, these are calculated in the iteration, does that matter?
    #	GammaxH - particle flux in x direction
    for k in range(0, nx):
            GammaxH[k] = Vth*np.sum(Vr2pidVr*(fH[:,:,k] @ (vx*dVx)))

    #	VxH - x velocity
    VxH = GammaxH / nH
    _VxH = VxH / Vth

    #	magnitude of random velocity at each mesh point
    vr2vx2_ran = np.zeros((nvr,nvx,nx))
    for i in range(0, nvr):
        for k in range(0, nx):
            vr2vx2_ran[i,:,k] = vr[i]**2 + (vx - _VxH[k])**2

    #	pH - pressure 
    for k in range(nx):
        pH[k] = ((mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*((vr2vx2_ran[:,:,k]*fH[:,:,k]) @ dVx))) / (3*CONST.Q)

    #	TH - temperature
    TH = pH/nH

    #	piH_xx
    for k in range(nx):
        KH.Output.piH_xx[k] = (((mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*(fH[:,:,k] @ (dVx*(vx - _VxH[k])**2)))) / CONST.Q) - pH[k]
    #	piH_yy
    for k in range(nx):
        KH.Output.piH_yy[k] = (((mu*CONST.H_MASS)*Vth2*0.5*np.sum((Vr2pidVr*(vr**2))*(fH[:,:,k] @ dVx))) / CONST.Q) - pH[k]
    #	piH_zz
    KH.Output.piH_zz = copy.copy(KH.Output.piH_yy)
    #	qxH
    for k in range(nx):
        qxH[k] = 0.5*(mu*CONST.H_MASS)*Vth3*np.sum(Vr2pidVr*((vr2vx2_ran[:,:,k]*fH[:,:,k]) @ (dVx*(vx - _VxH[k]))))

    #	C = RHS of Boltzman equation for total fH

    for k in range(nx):
        C = Vth*(Sn[:,:,k] + Beta_CX_sum[:,:,k] - alpha_c[:,:,k]*fH[:,:,k] + \
                 Omega_H_P[k]*MH_P_sum[:,:,k] + Omega_H_H2[k]*MH_H2_sum[:,:,k] + Omega_H_H[k]*MH_H_sum[:,:,k])
        # print("C", C.T)
        # input()
        QH[k] = 0.5*(mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*((vr2vx2_ran[:,:,k]*C) @ dVx))
        RxH[k] = (mu*CONST.H_MASS)*Vth*np.sum(Vr2pidVr*(C @ (dVx*(vx - _VxH[k]))))
        NetHSource[k] = np.sum(Vr2pidVr*(C @ dVx))
        Sion[k] = Vth*nH[k]*alpha_ion[k]
        KH.Output.SourceH[k] = np.sum(Vr2pidVr*(fSH[:,:,k] @ dVx))
        WallH[k] = np.sum(Vr2pidVr*((gamma_wall[:,:,k]*fH[:,:,k]) @ dVx))

        if Recomb:
            KH.Output.SRecomb[k] = Vth*ni[k]*Rec[k]
        else:
            KH.Output.SRecomb[k] = 0

        if H_P_CX:
            CCX = Vth*(Beta_CX_sum[:,:,k] - Alpha_CX[:,:,k]*fH[:,:,k])
            KH.Output.RxHCX[k] = (mu*CONST.H_MASS)*Vth*np.sum(Vr2pidVr*(CCX @ (dVx*(vx - _VxH[k]))))
            KH.Output.EHCX[k] = 0.5*(mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*((vr2vx2[:,:,k]*CCX) @ dVx))

        if H_H2_EL:
            CH_H2 = Vth*Omega_H_H2[k]*(MH_H2_sum[:,:,k] - fH[:,:,k])
            KH.Output.RxH2_H[k] = (mu*CONST.H_MASS)*Vth*np.sum(Vr2pidVr*(CH_H2 @ (dVx*(vx - _VxH[k]))))
            KH.Output.EH2_H[k] = 0.5*(mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*((vr2vx2[:,:,k]*CH_H2) @ dVx))

        if H_P_EL:
            CH_P = Vth*Omega_H_P[k]*(MH_P_sum[:,:,k] - fH[:,:,k])
            KH.Output.RxP_H[k] = (mu*CONST.H_MASS)*Vth*np.sum(Vr2pidVr*(CH_P @ (dVx*(vx - _VxH[k]))))
            KH.Output.EP_H[k] = 0.5*(mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*((vr2vx2[:,:,k]*CH_P) @ dVx))

        CW_H = -Vth*(gamma_wall[:,:,k]*fH[:,:,k])
        KH.Output.RxW_H[k] = (mu*CONST.H_MASS)*Vth*np.sum(Vr2pidVr*(CW_H @ (dVx*(vx - _VxH[k]))))
        KH.Output.EW_H[k] = 0.5*(mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*((vr2vx2[:,:,k]*CW_H) @ dVx))
        
        if H_H_EL:
            CH_H = Vth*Omega_H_H[k]*(MH_H_sum[:,:,k] - fH[:,:,k])
            for i in range(0, nvr):
                vr2_2vx_ran2[i,:] = vr[i]**2 - 2*((vx - _VxH[k])**2)
            KH.Output.Epara_PerpH_H[k] = -0.5*(mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*((vr2_2vx_ran2*CH_H) @ dVx))

    #	qxH_total
    qxH_total = (0.5*nH*(mu*CONST.H_MASS)*VxH*VxH + 2.5*pH*CONST.Q)*VxH + CONST.Q*KH.Output.piH_xx*VxH + qxH

    #	QH_total
    QH_total = QH + RxH*VxH + 0.5*(mu*CONST.H_MASS)*NetHSource*VxH*VxH

    #	Albedo
    gammax_plus = Vth*np.sum(Vr2pidVr*(fH[:,i_p[:,0],0] @ (vx[i_p[:,0]]*dVx[i_p[:,0]]))) #NOTE Had to reference i_p and i_n in a weird way, fix how they are called in the first place
    gammax_minus = Vth*np.sum(Vr2pidVr*(fH[:,i_n[:,0],0] @ (vx[i_n[:,0]]*dVx[i_n[:,0]]))) #This is awful and should not be allowed to remain
    if np.abs(gammax_plus) > 0:
        AlbedoH = -gammax_minus/gammax_plus

    #	Compute Mesh Errors

    mesh_error = np.zeros((nvr,nvx,nx))
    max_mesh_error = 0.0
    min_mesh_error = 0.0
    mtest = 5
    moment_error = np.zeros((nx,mtest))
    max_moment_error = np.zeros(mtest)
    C_error = np.zeros(nx)
    CX_error = np.zeros(nx)
    H_H_error = np.zeros((nx, 3))
    H_H2_error = np.zeros((nx, 3))
    H_P_error = np.zeros((nx, 3))
    max_H_H_error = np.zeros(3)
    max_H_H2_error = np.zeros(3)
    max_H_P_error = np.zeros(3)

    if compute_errors:
        if debrief > 1:
            print(prompt+'Computing Collision Operator, Mesh, and Moment Normalized Errors')
        NetHSource2 = KH.Output.SourceH + KH.Output.SRecomb - Sion - WallH
        for k in range(nx):
            C_error[k] = abs(NetHSource[k] - NetHSource2[k]) / max(abs(np.array([NetHSource[k], NetHSource2[k]])))

        #	Test conservation of particles for charge exchange operator
        if H_P_CX:
            for k in range(nx):
                CX_A = np.sum(Vr2pidVr*((Alpha_CX[:,:,k]*fH[:,:,k]) @ dVx))
                CX_B = np.sum(Vr2pidVr*(Beta_CX_sum[:,:,k] @ dVx))
                CX_error[k] = np.abs(CX_A - CX_B) / np.max(np.abs(np.array([CX_A, CX_B])))

        #	Test conservation of particles, x momentum, and total energy of elastic collision operators
        for m in range(0, 3):
            for k in range(0, nx):
                if m < 2:
                    TfH = np.sum(Vr2pidVr*(fH[:,:,k] @ (dVx*(vx**m))))
                else:
                    TfH = np.sum(Vr2pidVr*((vr2vx2[:,:,k]*fH[:,:,k]) @ dVx))

                if H_H_EL:
                    if m < 2:
                        TH_H = np.sum(Vr2pidVr*(MH_H_sum[:,:,k] @ (dVx*(vx**m))))
                    else:
                        TH_H = np.sum(Vr2pidVr*((vr2vx2[:,:,k]*MH_H_sum[:,:,k]) @ dVx))
                    H_H_error[k,m] = np.abs(TfH - TH_H) / np.max(np.abs(np.array([TfH, TH_H])))
                
                if H_H2_EL:
                    if m < 2:
                        TH_H2 = np.sum(Vr2pidVr*(MH_H2_sum[:,:,k] @ (dVx*(vx**m))))
                    else:
                        TH_H2 = np.sum(Vr2pidVr*((vr2vx2[:,:,k]*MH_H2_sum[:,:,k]) @ dVx))
                    H_H2_error[k,m] = np.abs(TfH - TH_H2) / np.max(np.abs(np.array([TfH, TH_H2])))

                if H_P_EL:
                    if m < 2:
                        TH_P = np.sum(Vr2pidVr*(MH_P_sum[:,:,k] @ (dVx*(vx**m))))
                    else:
                        TH_P = np.sum(Vr2pidVr*((vr2vx2[:,:,k]*MH_P_sum[:,:,k]) @ dVx))
                    H_P_error[k,m] = np.abs(TfH - TH_P) / np.max(np.abs(np.array([TfH, TH_P])))

            max_H_H_error[m] = np.max(H_H_error[:,m])
            max_H_H2_error[m] = np.max(H_H2_error[:,m])
            max_H_P_error[m] = np.max(H_P_error[:,m])

        if CI_Test:
            #	Compute Momentum transfer rate via full collision integrals for charge exchange and 
            #		mixed elastic scattering.
            #		Then compute error between this and actual momentum transfer 
            #		resulting from CX and BKG (elastic) models.

            if H_P_CX: # P -> H charge exchange momentum transfer via full collision integral
                print(prompt, 'Computing P -> H2 Charge Exchange Momentum Transfer')
                _Sig = np.zeros((nvr*nvx*nvr*nvx,ntheta))
                _Sig[:] = (v_v*sigma_cx_h0(v_v2*(0.5*CONST.H_MASS*Vth2 / CONST.Q))).reshape(_Sig.shape, order='F')
                SIG_VX_CX = np.zeros((nvr*nvx,nvr*nvx))
                SIG_VX_CX[:] = (Vr2pidVrdVx*vx_vx*((_Sig @ dTheta).reshape(vx_vx.shape, order='F'))).reshape(SIG_VX_CX.shape, order='F')
                alpha_vx_cx = np.zeros((nvr,nvx,nx))

                for k in range(0, nx):
                    Work[:] = (fi_hat[:,:,k]*ni[k]).reshape(Work.shape, order='F')
                    alpha_vx_cx[:,:,k] = (SIG_VX_CX @ Work).reshape(alpha_vx_cx[:,:,k].shape, order='F')

                for k in range(0, nx):
                    RxCI_CX[k] = -(mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*((alpha_vx_cx[:,:,k]*fH[:,:,k]) @ dVx))

                norm = np.max(np.abs(np.array([KH.Output.RxHCX, RxCI_CX])))
                for k in range(0, nx):
                    CI_CX_error[k] = np.abs(KH.Output.RxHCX[k] - RxCI_CX[k]) / norm

                print(prompt,'Maximum normalized momentum transfer error in CX collision operator: ', sval(np.max(CI_CX_error)))

            if H_P_EL: # P -> H momentum transfer via full collision integral
                for k in range(0, nx):
                    RxCI_P_H[k] = -(1/2)*(mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*((Alpha_H_P[:,:,k]*fH[:,:,k]) @ dVx))

                norm = np.max(np.abs(np.array([KH.Output.RxP_H, RxCI_P_H])))
                for k in range(0, nx):
                    CI_P_H_error[k] = np.abs(KH.Output.RxP_H[k] - RxCI_P_H[k]) / norm 

                print(prompt, 'Maximum normalized momentum transfer error in P -> H elastic BKG collision operator: ', sval(np.max(CI_P_H_error)))

            if H_H2_EL: # H2 -> H momentum transfer via full collision integral
                for k in range(0, nx):
                    RxCI_H2_H[k] = -(2/3)*(mu*CONST.H_MASS)*Vth2*np.sum(Vr2pidVr*((Alpha_H_H2[:,:,k]*fH[:,:,k]) @ dVx))
                
                norm = np.max(np.abs(np.array([KH.Output.RxH2_H, RxCI_H2_H])))
                for k in range(0, nx):
                    CI_H2_H_error[k] = np.abs(KH.Output.RxH2_H[k] - RxCI_H2_H[k])/norm
                
                print(prompt, 'Maximum normalized momentum transfer error in H2 -> H elastic BKG collision operator: ', sval(np.max(CI_H2_H_error)))

            if H_H_EL: # H -> H perp/parallel energy transfer via full collision integral
                for k in range(0, nx):
                    Work[:] = fH[:,:,k].reshape(Work.shape, order='F')
                    Alpha_H_H[:] = (SIG_H_H @ Work).reshape(Alpha_H_H.shape, order='F')
                    Epara_Perp_CI[k] = 0.5*(mu*CONST.H_MASS)*Vth3*np.sum(Vr2pidVr*((Alpha_H_H*fH[:,:,k]) @ dVx)) 
                
                norm = np.max(np.abs(np.array([KH.Output.Epara_PerpH_H, Epara_Perp_CI])))
                for k in range(0, nx):
                    CI_H_H_error[k] = np.abs(KH.Output.Epara_PerpH_H[k] - Epara_Perp_CI[k]) / norm 
                
                print(prompt, 'Maximum normalized perp/parallel energy transfer error in H -> H elastic BKG collision operator: ', sval(np.max(CI_H_H_error)))

        #	Mesh Point Error based on fH satisfying Boltzmann equation

        T1 = np.zeros((nvr,nvx,nx))
        T2 = np.zeros((nvr,nvx,nx))
        T3 = np.zeros((nvr,nvx,nx))
        T4 = np.zeros((nvr,nvx,nx))
        T5 = np.zeros((nvr,nvx,nx))
        for k in range(0, nx-1):
            for j in range(0, nvx):
                T1[:,j,k] = 2*vx[j]*(fH[:,j,k+1] - fH[:,j,k]) / (x[k+1] - x[k]) 
            T2[:,:,k] = (Sn[:,:,k+1] + Sn[:,:,k])
            T3[:,:,k] = Beta_CX_sum[:,:,k+1] + Beta_CX_sum[:,:,k]
            T4[:,:,k] = alpha_c[:,:,k+1]*fH[:,:,k+1] + alpha_c[:,:,k]*fH[:,:,k]
            T5[:,:,k] = Omega_H_P[k+1]*MH_P_sum[:,:,k+1] + Omega_H_H2[k+1]*MH_H2_sum[:,:,k+1] + Omega_H_H[k+1]*MH_H_sum[:,:,k+1] + \
                    Omega_H_P[k]*MH_P_sum[:,:,k] + Omega_H_H2[k]*MH_H2_sum[:,:,k] + Omega_H_H[k]*MH_H_sum[:,:,k]
            mesh_error[:,:,k] = np.abs(T1[:,:,k] - T2[:,:,k] - T3[:,:,k] + T4[:,:,k] - T5[:,:,k]) / \
                                np.max(np.abs(np.array([T1[:,:,k], T2[:,:,k], T3[:,:,k], T4[:,:,k], T5[:,:,k]])))
        ave_mesh_error = np.sum(mesh_error) / np.size(mesh_error)
        max_mesh_error = np.max(mesh_error)
        min_mesh_error = np.min(mesh_error[:,:,0:nx-1])

        #	Moment Error
        for m in range(0, mtest):
            for k in range(0, nx-1):
                MT1 = np.sum(Vr2pidVr*(T1[:,:,k] @ (dVx*(vx**m))))
                MT2 = np.sum(Vr2pidVr*(T2[:,:,k] @ (dVx*(vx**m))))
                MT3 = np.sum(Vr2pidVr*(T3[:,:,k] @ (dVx*(vx**m))))
                MT4 = np.sum(Vr2pidVr*(T4[:,:,k] @ (dVx*(vx**m))))
                MT5 = np.sum(Vr2pidVr*(T5[:,:,k] @ (dVx*(vx**m))))
                #NOTE This is correct for the original code, but is it correct mathematically?
                moment_error[k,m] = np.abs(MT1 - MT2 - MT3 + MT4 - MT5) / np.max(np.abs(np.array([MT1, MT2, MT3, MT4, MT5])))
            max_moment_error[m] = np.max(moment_error[:,m])

        #	Compute error in qxH_total

        #		qxH_total2 total neutral heat flux profile (watts m^-2)
        #			This is the total heat flux transported by the neutrals
        #			computed in a different way from:

        #			qxH_total2(k)=vth3*total(Vr2pidVr*((vr2vx2(*,*,k)*fH(*,*,k))#(Vx*dVx)))*0.5*(mu*mH)

        #			This should agree with qxH_total if the definitions of nH, pH, piH_xx,
        #			TH, VxH, and qxH are coded correctly.
        qxH_total2 = np.zeros(nx)
        for k in range(0, nx):
            qxH_total2[k] = 0.5*(mu*CONST.H_MASS)*Vth3*np.sum(Vr2pidVr*((vr2vx2[:,:,k]*fH[:,:,k]) @ (vx*dVx)))
        qxH_total_error = np.abs(qxH_total - qxH_total2) / np.max(np.abs(np.array([qxH_total, qxH_total2])))

        #	Compute error in QH_total
        Q1 = np.zeros(nx)
        Q2 = np.zeros(nx)
        QH_total_error = np.zeros(nx)
        for k in range(0, nx-1):
            Q1[k] = (qxH_total[k+1] - qxH_total[k]) / (x[k+1] - x[k])
            Q2[k] = 0.5*(QH_total[k+1] + QH_total[k])
        QH_total_error = np.abs(Q1 - Q2) / np.max(np.abs(np.array([Q1, Q2])))

        if debrief > 0:
            print(prompt+'Maximum particle convervation error of total collision operator: '+sval(max(C_error)))
            print(prompt+'Maximum H_P_CX  particle convervation error: '+sval(max(CX_error)))
            print(prompt+'Maximum H_H_EL  particle conservation error: '+sval(max_H_H_error[0]))
            print(prompt+'Maximum H_H_EL  x-momentum conservation error: '+sval(max_H_H_error[1]))
            print(prompt+'Maximum H_H_EL  total energy conservation error: '+sval(max_H_H_error[2]))
            print(prompt+'Maximum H_H2_EL particle conservation error: '+sval(max_H_H2_error[0]))
            print(prompt+'Maximum H_P_EL  particle conservation error: '+sval(max_H_P_error[0]))
            print(prompt+'Average mesh_error = '+str(ave_mesh_error))
            print(prompt+'Maximum mesh_error = '+str(max_mesh_error))
            for m in range(5):
                print(prompt+'Maximum fH vx^'+sval(m)+' moment error: '+sval(max_moment_error[m]))
            print(prompt+'Maximum qxH_total error = '+str(max(qxH_total_error)))
            print(prompt+'Maximum QH_total error = '+str(max(QH_total_error)))
            if debug > 0:
                input()

    # NOTE Add plotting once program is working

    #	Save input parameters in kinetic_H_input common block

    KH.Input.vx_s = vx
    KH.Input.vr_s = vr
    KH.Input.x_s = x
    KH.Input.Tnorm_s = Tnorm
    KH.Input.mu_s = mu
    KH.Input.Ti_s = Ti
    KH.Input.vxi_s = vxi
    KH.Input.Te_s = Te
    KH.Input.n_s = n
    KH.Input.vxi_s = vxi
    KH.Input.fHBC_s = fHBC
    KH.Input.GammaxHBC_s = GammaxHBC
    KH.Input.PipeDia_s = PipeDia
    KH.Input.fH2_s = fH2
    KH.Input.fSH_s = fSH
    KH.Input.nHP_s = nHP
    KH.Input.THP_s = THP
    KH.Input.fH_s = fH
    KH.Input.Simple_CX_s = Simple_CX
    KH.Input.JH_s = JH
    KH.Input.Collrad_s = Use_Collrad_Ionization
    KH.Input.Recomb_s = Recomb
    KH.Input.H_H_EL_s = H_H_EL
    KH.Input.H_P_EL_s = H_P_EL
    KH.Input.H_H2_EL_s = H_H2_EL
    KH.Input.H_P_CX_s = H_P_CX

    #	Save input parameters in kinetic_H_internal common block

    KH.Internal.vr2vx2 = vr2vx2
    KH.Internal.vr2vx_vxi2 = vr2vx_vxi2
    KH.Internal.fi_hat = fi_hat
    KH.Internal.ErelH_P = ErelH_P
    KH.Internal.Ti_mu = Ti_mu
    KH.Internal.ni = ni
    KH.Internal.sigv = sigv
    KH.Internal.alpha_ion = alpha_ion
    KH.Internal.v_v2 = v_v2
    KH.Internal.v_v = v_v
    KH.Internal.vr2_vx2 = vr2_vx2
    KH.Internal.vx_vx = vx_vx
    KH.Internal.Vr2pidVrdVx = Vr2pidVrdVx
    KH.Internal.SIG_CX = SIG_CX
    KH.Internal.SIG_H_H = SIG_H_H
    KH.Internal.SIG_H_H2 = SIG_H_H2
    KH.Internal.SIG_H_P = SIG_H_P
    KH.Internal.Alpha_CX = Alpha_CX
    KH.Internal.Alpha_H_H2 = Alpha_H_H2
    KH.Internal.Alpha_H_P = Alpha_H_P
    KH.Internal.MH_H_sum = MH_H_sum
    KH.Internal.Delta_nHs = Delta_nHs
    KH.Internal.Sn = Sn
    KH.Internal.Rec = Rec

    #	Save input parameters in kinetic_H_Moments common block

    KH.Moments.nH2 = nH2
    KH.Moments.VxH2 = vxH2
    KH.Moments.TH2 = TH2

    return fH,nH,GammaxH,VxH,pH,TH,qxH,qxH_total,NetHSource,Sion,QH,RxH,QH_total,AlbedoH,WallH,error
