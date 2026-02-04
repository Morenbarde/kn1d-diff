import numpy as np
from numpy.typing import NDArray
from scipy import interpolate
from dataclasses import dataclass
import os

from .create_shifted_maxwellian import create_shifted_maxwellian
from .make_dvr_dvx import VSpace_Differentials
from .utils import sval, interp_1d, get_config, make_json_compatible, sav_to_json
from .interp_fvrvxx import interp_fvrvxx
from .johnson_hinnov import Johnson_Hinnov
from .kinetic_mesh import KineticMesh
from .kinetic_h import KineticH
from .kinetic_h2 import KineticH2

from .common import constants as CONST


@dataclass
class KN1DResults():
    '''
    Data structure to hold and pass the results for kn1d
    '''
    # Molecular Info
    xH2: NDArray
    nH2: NDArray
    GammaxH2: NDArray
    TH2: NDArray
    qxH2_total: NDArray
    nHP: NDArray
    THP: NDArray
    SH: NDArray
    SP: NDArray

    # Atomic Info
    xH: NDArray
    nH: NDArray
    GammaxH: NDArray
    TH: NDArray
    qxH_total: NDArray
    NetHSource: NDArray
    Sion: NDArray
    QH_total: NDArray
    SideWallH: NDArray
    Lyman: NDArray
    Balmer: NDArray

    # Combined
    GammaHLim: NDArray

 

def kn1d(x, xlimiter, xsep, GaugeH2, mu, Ti, Te, n, vxi, LC, PipeDia,
         truncate = 1.0e-3, max_gen = 50,
         compute_errors = 0, debrief = 0,
         Hdebug = 0, Hdebrief = 0,
         H2debug = 0, H2debrief = 0, interp_debug = 0) -> dict:
    '''
    Computes the molecular and atomic neutral profiles for inputted profiles
    of Ti(x), Te(x), n(x), and molecular neutral pressure, GaugeH2, at the boundary using
    IDL routines Kinetic_H and Kinetic_H2. Molecular densities, ionization profiles,
    atomic densities and moments of the atomic distribution function, such as
    T0(x), Qin(x), qx0_total(x),... are returned. 

    It is assumed that molecular neutrals with temperature equal to the wall temperature
    (~ 1/40 eV) are attacking the plasma at x=x(0).

    History: First coding 5/1/2001  -  B. LaBombard

    Parameters
    ----------
        x : ndarray(nx)
            Cross-field coordinate (meters)
        xlimiter : float
            Cross-field coordinate of limiter edge (meters) (for graphic on plots)
        xsep : float
            Cross-field coordinate separatrix (meters) (for graphic on plots)
        GaugeH2	: float
            Molecular pressure (mtorr)
        mu : float
            1=hydrogen, 2=deuterium
        Ti : ndarray(nx)
            Ion temperature profile (eV)
        Te : ndarray(nx)
            Electron temperature profile (eV)
        n : ndarray(nx)
            Density profile (m^-3)
        vxi	: ndarray(nx)
            Plasma velocity profile [negative is towards 'wall' (m s^-1)]
        LC : ndarray(nx)
            Connection length (surface to surface) along field lines to nearest limiters (meters)
                Zero values of LC are treated as LC=infinity.
        PipeDia	: ndarray(nx)
            Effective pipe diameter (meters)
                This variable allows collisions with the 'side-walls' to be simulated.
                If this variable is undefined, then PipeDia set set to zero. Zero values
                of PipeDia are ignored (i.e., treated as an infinite diameter).
        truncate : float, default=1.0e-3
            Convergence threshold for generations
                fH and fH2 are refined by iteration via routines Kinetic_H2 and Kinetic_H
                until the maximum change in molecular neutral density (over its profile) normalized to 
                the maximum value of molecular density is less than this 
                value in a subsequent iteration.
        max_gen : int, default=50
            Maximum number of collision generations to try including before giving up.

    Returns
    -------
    KN1DResults

        Molecular info
            - xH2: ndarray(nxH2), Cross-field coordinate for molecular quantities (meters)
            - nH2: ndarray(nxH2), Neutral moleular density profile (m^-3)
            - GammaxH2: ndarray(nxH2), Neutral flux profile (# m^-2 s^-1)
            - TH2: ndarray(nxH2), Molecular neutral temperature profile (m^-3)
            - qxH2_total: ndarray(nxH2), Molecular neutral heat flux profile (watts m^-2)
            - nHP: ndarray(nxH2), Molecular ion density profile (m^-3)
            - THP: ndarray(nxH2), Molecular ion temperature profile (eV)
            - SH: ndarray(nxH2), Atomic source profile (m^-3 s^-1)
            - SP: ndarray(nxH2), Ion source profile (m^-3 s^-1)

        Atomic info
            - xH: ndarray(nxH), Cross-field coordinate for atomic quantities (meters)
            - nH: ndarray(nxH), Neutral atomic density profile (m^-3)
            - GammaxH: ndarray(nxH), Neutral flux profile (# m^-2 s^-1)
            - TH: ndarray(nxH), Atomic neutral temperature profile (m^-3)
            - qxH_total: ndarray(nxH), Atomic neutral heat flux profile (watts m^-2)
            - NetHSource: ndarray(nxH), Net source of atomic neutrals from molecular dissociation and recomb minus ionization (# m^-3) 
            - Sion: ndarray(nxH), Atomic ionization rate (# m^-3) 
            - QH_total: ndarray(nxH), Net rate of total energy transfer to atomic neutral species (watts m^-3)
            - SideWallH: ndarray(nxH), Atomic neutral sink rate arising from hitting the 'side walls' (m^-3 s^-1)
                    Unlike the molecules in Kinetic_H2, wall collisions result in the destruction of atoms.
                    This parameter is used to specify a resulting source of molecular
                    neutrals in Kinetic_H2. (molecular source = 2 times SideWallH)
            - Lyman: ndarray(nxH), Lyman-alpha emissivity (watts m^-3) using rate coefficients of L.C.Johnson and E. Hinnov
            - Balmer: ndarray(nxH), Balmer-alpha emissivity (watts m^-3) using rate coefficients of L.C.Johnson and E. Hinnov
    
        Combined
            - GammaHLim: float, 2*GammaxH2 + GammaxH at edge of limiter (# m^-2 s^-1)
    '''

    prompt = 'KN1D => '

    # --- Validate Config Options ---
    
    valid_ion_rates = ['collrad', 'jh', 'janev']
    ion_rate_option = get_config()['kinetic_h']['ion_rate']
    if ion_rate_option not in valid_ion_rates:
        raise Exception(prompt+"Invalid Ionization Rate Option used: '"+ion_rate_option+"', check config.json")

    
    # --- Generate Meshes ---

    # Determine optimized vr, vx, grid for kinetc_h2 (molecules, M)
    Eneut = np.array([0.003,0.01,0.03,0.1,0.3,1.0,3.0])
    fctr = 0.3
    if GaugeH2 > 15.0:
        fctr = fctr*15 / GaugeH2

    



    kh2_mesh = KineticMesh('h2', mu, x, Ti, Te, n, PipeDia, E0 = Eneut, fctr = fctr) 

 
    
    # Determine optimized vr, vx grid for kinetic_h (atoms, A)
    fctr = 0.3
    if GaugeH2 > 30.0 :
        fctr = fctr * 30 / GaugeH2

    # Generates Johnson_Hinnov class, Used in place of IDL version's JH_Coef Common block
    jh = Johnson_Hinnov()

    kh_mesh = KineticMesh('h', mu, x, Ti, Te, n, PipeDia, jh=jh, fctr=fctr)


    # --- Initialize variables ---
    
    # Initialize fH and fH2
    
    fH = np.zeros((kh_mesh.vr.size,kh_mesh.vx.size,kh_mesh.x.size))
    fH2 = np.zeros((kh2_mesh.vr.size,kh2_mesh.vx.size,kh2_mesh.x.size))
    nH2 = np.zeros(kh2_mesh.x.size)
    nHP = np.zeros(kh2_mesh.x.size)
    THP = np.zeros(kh2_mesh.x.size)
        
    # Directed random velocity of diatomic molecule
    v0_bar = np.sqrt((8.0*CONST.TWALL*CONST.Q) / (np.pi*2*mu*CONST.H_MASS))
    # Set up molecular flux BC from inputted neutral pressure
    ipM = np.where(kh2_mesh.vx > 0)

    # Convert pressure (mtorr) to molecular density and flux
    fh2BC = np.zeros((kh2_mesh.vr.size,kh2_mesh.vx.size), float)
    DensM = 3.537e19*GaugeH2
    GammaxH2BC = 0.25*DensM*v0_bar
    Tmaxwell = np.array([CONST.TWALL])
    vx_shift = np.array([0.0])
    Maxwell = create_shifted_maxwellian(kh2_mesh.vr, kh2_mesh.vx, Tmaxwell, vx_shift, mu, 2, kh2_mesh.Tnorm)
    fh2BC[:,ipM] = Maxwell[:,ipM,0]

    # Compute NuLoss (Cs/LC)
    Cs_LC = np.zeros(LC.size)
    for ii in range(LC.size):
        if LC[ii] > 0:
            Cs_LC[ii] = np.sqrt(CONST.Q*(Ti[ii] + Te[ii]) / (mu*CONST.H_MASS)) / LC[ii]
    NuLoss = interp_1d(x, Cs_LC, kh2_mesh.x)
    

    #  Compute first guess SpH2
    #_____________________________________________________________________________________________________________
    #   If plasma recycling accounts for molecular source, then SpH2 = 1/2 n Cs/LC (1/2 accounts for H2 versus H)
    #   But, allow for SpH2 to be proportional to this function:
    #      SpH2 = beta n Cs/LC 
    #   with beta being an adjustable parameter, set by achieving a net H flux of zero onto the wall.
    #   For first guess of beta, set the total molecular source according to the formula
    #
    # (See notes "Procedure to adjust the normalization of the molecular source at the 
    #   limiters (SpH2) to attain a net zero atom/molecule flux from wall")
    #
    #	Integral{SpH2}dx =  (2/3) GammaxH2BC = beta Integral{n Cs/LC}dx
    #______________________________________________________________________________________________________________

    SpH2_hat = interp_1d(x, n*Cs_LC, kh2_mesh.x, fill_value="extrapolate")

    SpH2_hat /= np.trapezoid(SpH2_hat, kh2_mesh.x)
    beta = (2/3)*GammaxH2BC
    SpH2 = beta*SpH2_hat
    SH2 = SpH2


    #   Interpolate for vxiM and vxiA

    interpfunc = interpolate.interp1d(x, vxi, fill_value="extrapolate")
    vxiM = interpfunc(kh2_mesh.x)
    vxiA = interpfunc(kh_mesh.x)


    # Compute mesh differentials

    vthM = np.sqrt(2*CONST.Q*kh2_mesh.Tnorm/(mu*CONST.H_MASS))
    kh2_differentials = VSpace_Differentials(kh2_mesh.vr, kh2_mesh.vx)

    #NOTE Used in gammalim calculation, will be needed later
    vthA = np.sqrt(2*CONST.Q*kh_mesh.Tnorm/(mu*CONST.H_MASS))
    kh_differentials = VSpace_Differentials(kh_mesh.vr, kh_mesh.vx)

    #  Test for v0_bar consistency in the numerics by computing it from a half maxwellian at the wall temperature

    nbarHMax = np.sum(kh2_differentials.dvr_vol*(fh2BC @ kh2_differentials.dvx))
    vbarM = 2*vthM*np.sum(kh2_differentials.dvr_vol*((fh2BC @ (kh2_mesh.vx*kh2_differentials.dvx))))/nbarHMax
    vbarM_error = abs(vbarM - v0_bar)/max(vbarM, v0_bar)

    vr2vx2_ran2 = np.zeros((kh2_mesh.vr.size,kh2_mesh.vx.size))

    mwell = Maxwell[:,:,0]

    nbarMax = np.sum(kh2_differentials.dvr_vol*(mwell @ kh2_differentials.dvx))
    UxMax = vthM*np.sum(kh2_differentials.dvr_vol*(mwell @ (kh2_mesh.vx*kh2_differentials.dvx)))/nbarMax

    for i in range(kh2_mesh.vr.size):
        vr2vx2_ran2[i,:] = kh2_mesh.vr[i]**2 + (kh2_mesh.vx - UxMax/vthM)**2
    TMax = 2*mu*CONST.H_MASS*(vthM**2)*np.sum(kh2_differentials.dvr_vol*((vr2vx2_ran2*mwell) @ kh2_differentials.dvx))/(3*CONST.Q*nbarMax)

    UxHMax = vthM*np.sum(kh2_differentials.dvr_vol*(fh2BC @ (kh2_mesh.vx*kh2_differentials.dvx)))/nbarHMax
    for i in range(kh2_mesh.vr.size):
        vr2vx2_ran2[i,:] = kh2_mesh.vr[i]**2 + (kh2_mesh.vx - UxHMax/vthM)**2
    THMax = (2*mu*CONST.H_MASS)*(vthM**2)*np.sum(kh2_differentials.dvr_vol*((vr2vx2_ran2*fh2BC) @ kh2_differentials.dvx))/(3*CONST.Q*nbarHMax)

    if compute_errors and debrief:
        print(prompt+'VbarM_error: '+sval(vbarM_error))
        print(prompt+'TWall Maxwellian: '+sval(TMax))
        print(prompt+'TWall Half Maxwellian: '+sval(THMax))


    # --- Setup Procedure Classes ---

    GammaxHBC = 0
    fHBC = np.zeros((kh_mesh.vr.size,kh_mesh.vx.size))

    # file = 'kinetic_h_in.npz'
    # print("Saving to file: " + file)
    # np.savez("kn1ddiff/test/init_kinetic_h/"+file, mu=mu, vxiA=vxiA, fHBC=fHBC, GammaxHBC=GammaxHBC)
    # input()
    kinetic_h = KineticH(kh_mesh, mu, vxiA, fHBC, GammaxHBC, jh=jh,
                         ni_correct=True, truncate=truncate, max_gen=max_gen, 
                         compute_errors=compute_errors, debrief=Hdebrief, debug=Hdebug)
    # file = 'kinetic_h_params.npz'
    # print("Saving to file: " + file)
    # np.savez("kn1ddiff/test/init_kinetic_h/"+file, mu=kinetic_h.mu, vxi=kinetic_h.vxi, fHBC=kinetic_h.fHBC, 
    #          GammaxHBC=kinetic_h.GammaxHBC, nvr=kinetic_h.nvr, nvx=kinetic_h.nvx, nx=kinetic_h.nx, vx_neg=kinetic_h.vx_neg,
    #          vx_pos=kinetic_h.vx_pos, vx_zero=kinetic_h.vx_zero, vth=kinetic_h.vth, vr2_2vx2_2D=kinetic_h.vr2_2vx2_2D,
    #          dvr_vol=kinetic_h.dvr_vol, dvx=kinetic_h.dvx, fHBC_input=kinetic_h.fHBC_input)
    # input()

    # file = 'kinetic_h_internal.json'
    # data = kinetic_h.Internal
    # print("Saving to file: " + file)
    # sav_data = {'vr2vx2' : data.vr2vx2,
    #             'vr2vx_vxi2' : data.vr2vx_vxi2,
    #             'fi_hat' : data.fi_hat,
    #             'ErelH_P' : data.ErelH_P,
    #             'Ti_mu' : data.Ti_mu,
    #             'ni' : data.ni,
    #             'sigv' : data.sigv,
    #             'alpha_ion' : data.alpha_ion,
    #             'v_v2' : data.v_v2,
    #             'v_v' : data.v_v,
    #             'vr2_vx2' : data.vr2_vx2,
    #             'vx_vx' : data.vx_vx,

    #             'Vr2pidVrdVx' : data.Vr2pidVrdVx,
    #             'SIG_CX' : data.SIG_CX,
    #             'SIG_H_H' : data.SIG_H_H,
    #             'SIG_H_H2' : data.SIG_H_H2,
    #             'SIG_H_P' : data.SIG_H_P,
    #             'Alpha_CX' : data.Alpha_CX,
    #             'Alpha_H_H2' : data.Alpha_H_H2,
    #             'Alpha_H_P' : data.Alpha_H_P,
    #             'MH_H_sum' : data.MH_H_sum,
    #             'Delta_nHs' : data.Delta_nHs,
    #             'Sn' : data.Sn,
    #             'Rec' : data.Rec}
    # sav_data = make_json_compatible(sav_data)
    # sav_to_json("kn1ddiff/test/init_kinetic_h/"+file, sav_data)
    # input()
    
    kinetic_h2 = KineticH2(kh2_mesh, mu, vxiM, fh2BC, GammaxH2BC, NuLoss, SH2,
                            compute_h_source=True, ni_correct=True, truncate=truncate, max_gen=max_gen, 
                            compute_errors=compute_errors, debrief=H2debrief, debug=H2debug)


    # --- Begin Iteration ---

    print(prompt+"Satisfaction condition: ", truncate)


    iter = 0
    EH_hist = np.array([0.0])
    SI_hist = np.array([0.0])
    while True:
        # Iterates through solving fh and fh2 until they satisfy boltzmans equation

        iter += 1
        if debrief:
            print(prompt+'fH/fH2 Iteration: '+sval(iter))
        nH2_saved = nH2

        # interpolate fH data onto H2 mesh: fH -> fHM
        do_warn = 5e-3
        fHM = interp_fvrvxx(fH, kh_mesh, kh2_mesh, do_warn=do_warn, debug=interp_debug)


        # --- Run kinetic_h2 ---

        kh2_results = kinetic_h2.run_procedure(fHM, SH2, fH2, nHP, THP)
        
        fH2 = kh2_results.fH2
        nHP = kh2_results.nHP
        THP = kh2_results.THP
        nH2 = kh2_results.nH2


        # Interpolate H2 data onto H mesh: fH2 -> fH2A, fSH -> fSHA, nHP -> nHPA, THP -> THPA
        do_warn = 5.0E-3
        fH2A = interp_fvrvxx(fH2, kh2_mesh, kh_mesh, do_warn=do_warn, debug=interp_debug) 
        fSHA = interp_fvrvxx(kh2_results.fSH, kh2_mesh, kh_mesh, do_warn=do_warn, debug=interp_debug) #NOTE return value here not correct, see _Wxa calculation, set debug_flag

        nHPA = np.interp(kh_mesh.x, kh2_mesh.x, nHP, left=0, right=0)
        THPA = np.interp(kh_mesh.x, kh2_mesh.x, THP, left=0, right=0)


        # --- Run kinetic_h ---

        kh_results = kinetic_h.run_procedure(fH2A, fSHA, fH, nHPA, THPA)
        
        fH = kh_results.fH


        # Interpolate SideWallH data onto H2 mesh: SideWallH -> SideWallHM
        SideWallHM = np.interp(kh2_mesh.x, kh_mesh.x, kh_results.SideWallH, left=0, right=0)

        # Adjust SpH2 to achieve net zero hydrogen atom/molecule flux from wall
        # (See notes "Procedure to adjust the normalization of the molecular source at the 
        # limiters (SpH2) to attain a net zero atom/molecule flux from wall")

        # Compute SI, GammaH2Wall_minus, and GammaHWall_minus
        SI = np.trapezoid(SpH2, kh2_mesh.x)
        SwallI = np.trapezoid(0.5*SideWallHM, kh2_mesh.x)
        GammaH2Wall_minus = kh2_results.AlbedoH2*GammaxH2BC
        GammaHWall_minus = -kh_results.GammaxH[0]

        # Compute Epsilon and alphaplus1RH0Dis
        Epsilon = 2*GammaH2Wall_minus / (SI+SwallI)
        alphaplus1RH0Dis = GammaHWall_minus / ((1 - 0.5*Epsilon)*(SI + SwallI) + GammaxH2BC)

        # Compute flux error, EH, and dEHdSI
        EH = 2*kh2_results.GammaxH2[0] - GammaHWall_minus
        dEHdSI = -Epsilon - alphaplus1RH0Dis*(1 - 0.5*Epsilon)

        # Option: print normalized flux error
        nEH = np.abs(EH) / np.max(np.abs(np.array([2*kh2_results.GammaxH2[0], GammaHWall_minus] )))
        if debrief and compute_errors:
            print(prompt, 'Normalized Hydrogen Flux Error: ', sval(nEH))
        
        # Compute Adjustment 
        Delta_SI = -EH/dEHdSI
        SI = SI + Delta_SI

        # Rescale SpH2 to have new integral value, SI
        SpH2 = SI*SpH2_hat
        EH_hist = np.append(EH_hist, EH)
        SI_hist = np.append(SI_hist, SI)

        # Set total H2 source
        SH2 = SpH2 + 0.5*SideWallHM

        if compute_errors:
            _RxH_H2 = np.interp(kh_mesh.x, kh2_mesh.x, kinetic_h2.Output.RxH_H2, left=0, right=0)
            DRx = _RxH_H2 + kinetic_h.Output.RxH2_H
            nDRx = np.max(np.abs(DRx)) / np.max(np.abs(np.array([_RxH_H2, kinetic_h.Output.RxH2_H])))
            if debrief:
                print(prompt, 'Normalized H2 <-> H Momentum Transfer Error: ', sval(nDRx))
                
        
        Delta_nH2 = np.abs(kh2_results.nH2 - nH2_saved)
        nDelta_nH2 = np.max(Delta_nH2/np.max(kh2_results.nH2))
        if debrief: 
            print(prompt, 'Maximum Normalized change in nH2: ', sval(nDelta_nH2))

        if nDelta_nH2 <= truncate:
            # Stop Iteration
            break
    
    # --- End Iteration ---

    #NOTE Add gammaHLim
    gamma_h2 = np.interp(kh_mesh.x, kh2_mesh.x, kh2_results.GammaxH2)
    gam = 2*gamma_h2 + kh_results.GammaxH
    GammaHLim = interp_1d(kh_mesh.x, gam, xlimiter)


    
    # --- Compute Lyman and Balmer Alpha ---

    Lyman = jh.lyman_alpha(kh_mesh.ne, kh_mesh.Te, kh_results.nH, no_null=1)
    Balmer = jh.balmer_alpha(kh_mesh.ne, kh_mesh.Te, kh_results.nH, no_null=1)
    # Lyman = lyman_alpha(kh_mesh.ne, kh_mesh.Te, nH, jh_coefficients, no_null = 1) #NOTE Not Working Yet
    # Balmer = balmer_alpha(kh_mesh.ne, kh_mesh.Te, nH, jh_coefficients, no_null = 1) #NOTE Not Working Yet


    # --- Store Results ---

    # Store Outputs
    output_dir = 'Results/'
    output_file = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = output_dir+output_file
    print(prompt, "Saving files to", output_path+".npz")
    np.savez(output_path,
             xH2=kh2_mesh.x,
             nH2=kh2_results.nH2,
             GammaxH2=kh2_results.GammaxH2,
             TH2=kh2_results.TH2,
             qxH2_total=kh2_results.qxH2_total,
             nHP=kh2_results.nHP,
             THP=kh2_results.THP,
             SH=kh2_results.SH,
             SP=kh2_results.SP,

             xH=kh_mesh.x,
             nH=kh_results.nH,
             GammaxH=kh_results.GammaxH,
             TH=kh_results.TH,
             qxH_total=kh_results.qxH_total,
             NetHSource=kh_results.NetHSource,
             Sion=kh_results.Sion,
             QH_total=kh_results.QH_total,
             SideWallH=kh_results.SideWallH,
             Lyman=Lyman,
             Balmer=Balmer,
             GammaHLim=GammaHLim)


    # Format Results into Dataclass
    results = KN1DResults(kh2_mesh.x, 
                          kh2_results.nH2, 
                          kh2_results.GammaxH2, 
                          kh2_results.TH2, 
                          kh2_results.qxH2_total,
                          kh2_results.nHP,
                          kh2_results.THP,
                          kh2_results.SH,
                          kh2_results.SP,

                          kh_mesh.x,
                          kh_results.nH,
                          kh_results.GammaxH,
                          kh_results.TH,
                          kh_results.qxH_total,
                          kh_results.NetHSource,
                          kh_results.Sion,
                          kh_results.QH_total,
                          kh_results.SideWallH,
                          Lyman,
                          Balmer,

                          GammaHLim)

    return results
