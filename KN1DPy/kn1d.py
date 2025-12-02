import numpy as np 
from numpy.typing import NDArray
import os
from scipy import interpolate

from .utils import sav_read, nc_read
from .create_shifted_maxwellian import create_shifted_maxwellian
from .make_dvr_dvx import VSpace_Differentials
from .utils import sval, interp_1d
from .interp_fvrvxx import interp_fvrvxx
from .kinetic_mesh import KineticMesh
from .kinetic_h import kinetic_h 
from .kinetic_h2 import kinetic_h2
from .jh_related.lyman_alpha import lyman_alpha
from .jh_related.balmer_alpha import balmer_alpha

from .common import constants as CONST
from .common.Kinetic_H2 import Kinetic_H2_Common
from .common.Kinetic_H import Kinetic_H_Common
from .common.JH_Coef import JH_Coef


#   Computes the molecular and atomic neutral profiles for inputted profiles
# of Ti(x), Te(x), n(x), and molecular neutral pressure, GaugeH2, at the boundary using
# IDL routines Kinetic_H and Kinetic_H2. Molecular densities, ionization profiles,
# atomic densities and moments of the atomic distribution function, such as
# T0(x), Qin(x), qx0_total(x),... are returned. 

#   It is assumed that molecular neutrals with temperature equal to the wall temperature
# (~ 1/40 eV) are attacking the plasma at x=x(0).
#
# History: First coding 5/1/2001  -  B. LaBombard
 

def kn1d(x, xlimiter, xsep, GaugeH2, mu, Ti, Te, n, vxi, LC, PipeDia, \
         truncate = 1.0e-3, max_gen = 50, \
         error = 0, compute_errors = 0, plot = 0, debug = 0, debrief = 0, pause = 0, \
         Hplot = 0, Hdebug = 0, Hdebrief = 0, Hpause = 0, \
         H2plot = 0, H2debug = 0, H2debrief = 0, H2pause = 0, interp_debug = 0) -> dict:

    # Input: 
    #	x	        - fltarr(nx), cross-field coordinate (meters)
    #   xlimiter    - float, cross-field coordinate of limiter edge (meters) (for graphic on plots)
    #	xsep	    - float, cross-field coordinate separatrix (meters) (for graphic on plots)
    #	GaugeH2	    - float, Molecular pressure (mtorr)
    #	mu	        - float, 1=hydrogen, 2=deuterium
    #	Ti	        - fltarr(nx), ion temperature profile (eV)
    #	Te	        - fltarr(nx), electron temperature profile (eV)
    #	n	        - fltarr(nx), density profile (m^-3)
    #	vxi	        - fltarr(nx), plasma velocity profile [negative is towards 'wall' (m s^-1)]
    #	LC	        - fltarr(nx), connection length (surface to surface) along field lines to nearest limiters (meters)
    #                   Zero values of LC are treated as LC=infinity.
    #   PipeDia	    - fltarr(nx), effective pipe diameter (meters)
    #                   This variable allows collisions with the 'side-walls' to be simulated.
    #                   If this variable is undefined, then PipeDia set set to zero. Zero values
    #                   of PipeDia are ignored (i.e., treated as an infinite diameter).
    #
    # Keyword Input:
    #   truncate	- float, this parameter is also passed to Kinetic_H and Kinetic_H2.
    #                   fH and fH2 are refined by iteration via routines Kinetic_H2 and Kinetic_H
    #		            until the maximum change in molecular neutral density (over its profile) normalized to 
    #		            the maximum value of molecular density is less than this 
    #	    	        value in a subsequent iteration. Default value is 1.0e-3
    #   max_gen     - integer, maximum number of collision generations to try including before giving up.
    #   refine      - NOTE Not Currently Implemented. if set, then use previously computed atomic and molecular distribution functions
    #		            stored in internal common block (if any) or from FILE (see below) as the initial 'seed' value'
    #   file        - NOTE Not Currently Implemented. string, if not null, then read in 'file'.kn1d_mesh save set and compare contents
    #                   to the present input parameters and computational mesh. If these are the same
    #		            then read results from previous run in 'file'.kn1d_H2 and 'file'.kn1d_H.
    #   Newfile     - NOTE Not Currently Implemented. if set, then do not generate an error and exit if 'file'.KN1D_mesh or 'file'.KN1D_H2
    #                   or 'file'.KN1D_H do not exist or differ from the present input parameters. Instead, write 
    #                   new mesh and output files on exiting.
    #   ReadInput   - NOTE Not Currently Implemented. if set, then reset all input variables to that contained in 'file'.KN1D_input

    # Output:
    #   Molecular info
    #       xH2	        - fltarr(nxH2), cross-field coordinate for molecular quantities (meters)
    #       nH2	        - fltarr(nxH2), neutral moleular density profile (m^-3)
    #       GammaxH2    - fltarr(nxH2), neutral flux profile (# m^-2 s^-1)
    #       TH2	        - fltarr(nxH2), molecular neutral temperature profile (m^-3)
    #       qxH2_total	- fltarr(nxH2), molecular neutral heat flux profile (watts m^-2)
    #       nHP	        - fltarr(nxH2), molecular ion density profile (m^-3)
    #       THP	        - fltarr(nxH2), molecular ion temperature profile (eV)
    #       SH	        - fltarr(nxH2), atomic source profile (m^-3 s^-1)
    #       SP	        - fltarr(nxH2), ion source profile (m^-3 s^-1)
    #
    #   Atomic info
    #       xH	        - fltarr(nxH), cross-field coordinate for atomic quantities (meters)
    #       nH	        - fltarr(nxH), neutral atomic density profile (m^-3)
    #       GammaxH 	- fltarr(nxH), neutral flux profile (# m^-2 s^-1)
    #       TH	        - fltarr(nxH), atomic neutral temperature profile (m^-3)
    #       qxH_total	- fltarr(nxH), atomic neutral heat flux profile (watts m^-2)
    #       NetHSource	- fltarr(nxH), net source of atomic neutrals from molecular dissociation and recomb minus ionization (# m^-3) 
    #	    Sion	    - fltarr(nxH), atomic ionization rate (# m^-3) 
    #	    QH_total    - fltarr(nxH), net rate of total energy transfer to atomic neutral species (watts m^-3)
    #       SideWallH	- fltarr(nxH), atomic neutral sink rate arising from hitting the 'side walls' (m^-3 s^-1)
    #	                    Unlike the molecules in Kinetic_H2, wall collisions result in the destruction of atoms.
    #                       This parameter is used to specify a resulting source of molecular
    #                       neutrals in Kinetic_H2. (molecular source = 2 times SideWallH)
    #	    Lyman       - fltarr(nxH), Lyman-alpha emissivity (watts m^-3) using rate coefficients of L.C.Johnson and E. Hinnov
    #	    Balmer      - fltarr(nxH), Balmer-alpha emissivity (watts m^-3) using rate coefficients of L.C.Johnson and E. Hinnov


    x = np.array(x, dtype=np.float64)

    # directed random velocity of diatomic molecule
    v0_bar = np.sqrt((8.0*CONST.TWALL*CONST.Q) / (np.pi*2*mu*CONST.H_MASS))

    prompt = 'KN1D => '


    #Generates JH_Coef class, Used in place of IDL version's JH_Coef Common block
    jh_coefficients = JH_Coef()
        
    # Option: Read input parameters stored in file from previous run
    #   NOTE Removed, consider implementing later
    


    # determine optimized vr, vx, grid for kinetc_h2 (molecules, M)
    Eneut = np.array([0.003,0.01,0.03,0.1,0.3,1.0,3.0])
    fctr = 0.3
    if GaugeH2 > 15.0:
        fctr = fctr*15 / GaugeH2

    kh2_mesh = KineticMesh('h2', mu, x, Ti, Te, n, PipeDia, E0 = Eneut, fctr = fctr) 
    
    # determine optimized vr, vx grid for kinetic_h (atoms, A)
    fctr = 0.3
    if GaugeH2 > 30.0 :
        fctr = fctr * 30 / GaugeH2

    kh_mesh = KineticMesh('h', mu, x, Ti, Te, n, PipeDia, jh_coeffs = jh_coefficients, fctr=fctr)


    #   Set up molecular flux BC from inputted neutral pressure

    ipM = np.where(kh2_mesh.vx > 0)
    
    #  Initialize fH and fH2 (these may be over-written by data from and old run below)
    
    # NOTE Refine Deleted, possibly re-add later
    fH = np.zeros((kh_mesh.vr.size,kh_mesh.vx.size,kh_mesh.x.size))
    fH2 = np.zeros((kh2_mesh.vr.size,kh2_mesh.vx.size,kh2_mesh.x.size))
    nH2 = np.zeros(kh2_mesh.x.size)
    nHP = np.zeros(kh2_mesh.x.size)
    THP = np.zeros(kh2_mesh.x.size)
        

    #   Convert pressure (mtorr) to molecular density and flux

    fh2BC = np.zeros((kh2_mesh.vr.size,kh2_mesh.vx.size), float)
    DensM = 3.537e19*GaugeH2
    GammaxH2BC = 0.25*DensM*v0_bar
    Tmaxwell = np.array([CONST.TWALL])
    vx_shift = np.array([0.0])
    Maxwell = create_shifted_maxwellian(kh2_mesh.vr, kh2_mesh.vx, Tmaxwell, vx_shift, mu, 2, kh2_mesh.Tnorm)
    fh2BC[:,ipM] = Maxwell[:,ipM,0]


    # Compute NuLoss:
        # NuLoss = Cs/LC
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

    #interpfunc = interpolate.interp1d(x,vxi,fill_value="extrapolate") #NOTE Is this Needed?
    vxiA = interpfunc(kh_mesh.x)
    # print("vxiM", vxiM)
    # print("vxiA", vxiA)
    # input()

    iter = 0
    EH_hist = np.array([0.0])
    SI_hist = np.array([0.0])
    oldrun = 0

    #   Option: Read results from previous run
        #   NOTE Implement Later

    #   Starting back at line 378 from IDL code
    #   Test for v0_bar consistency in the numerics by computing it from a half maxwellian at the wall temperature

    vthM = np.sqrt(2*CONST.Q*kh2_mesh.Tnorm/(mu*CONST.H_MASS))
    kh2_differentials = VSpace_Differentials(kh2_mesh.vr, kh2_mesh.vx)
    kh2_differentials.dvx

    #NOTE Used in gammalim calculation, will be needed later
    vthA = np.sqrt(2*CONST.Q*kh_mesh.Tnorm/(mu*CONST.H_MASS))
    kh_differentials = VSpace_Differentials(kh_mesh.vr, kh_mesh.vx)


    nbarHMax = np.sum(kh2_differentials.dvr_vol*(fh2BC @ kh2_differentials.dvx))
    vbarM = 2*vthM*np.sum(kh2_differentials.dvr_vol*((fh2BC @ (kh2_mesh.vx*kh2_differentials.dvx))))/nbarHMax
    vbarM_error = abs(vbarM - v0_bar)/max(vbarM, v0_bar)
    # print("nbarHMax", nbarHMax)
    # print("vbarM", vbarM)
    # print("vbarM_error", vbarM_error)
    # input()

    vr2vx2_ran2 = np.zeros((kh2_mesh.vr.size,kh2_mesh.vx.size)) # fixed indexing - GG

    mwell = Maxwell[:,:,0] #  variable named 'Max' in original code; changed here to avoid sharing name with built in function
    # print("mwell", mwell.T)
    # input()

    nbarMax = np.sum(kh2_differentials.dvr_vol*(mwell @ kh2_differentials.dvx))
    UxMax = vthM*np.sum(kh2_differentials.dvr_vol*(mwell @ (kh2_mesh.vx*kh2_differentials.dvx)))/nbarMax
    # print("1", kh2_mesh.vx)
    # print("2", dVxM)
    # print("3", vthM)
    # print("4", Vr2pidVrM)
    for i in range(kh2_mesh.vr.size):
        vr2vx2_ran2[i,:] = kh2_mesh.vr[i]**2 + (kh2_mesh.vx - UxMax/vthM)**2
    TMax = 2*mu*CONST.H_MASS*(vthM**2)*np.sum(kh2_differentials.dvr_vol*((vr2vx2_ran2*mwell) @ kh2_differentials.dvx))/(3*CONST.Q*nbarMax)
    # print("nbarMax", nbarMax)
    # print("UxMax", UxMax)
    # print("vr2vx2_ran2", vr2vx2_ran2.T)
    # print("Tmax", TMax)
    # input()

    UxHMax = vthM*np.sum(kh2_differentials.dvr_vol*(fh2BC @ (kh2_mesh.vx*kh2_differentials.dvx)))/nbarHMax
    for i in range(kh2_mesh.vr.size):
        vr2vx2_ran2[i,:] = kh2_mesh.vr[i]**2 + (kh2_mesh.vx - UxHMax/vthM)**2
    THMax = (2*mu*CONST.H_MASS)*(vthM**2)*np.sum(kh2_differentials.dvr_vol*((vr2vx2_ran2*fh2BC) @ kh2_differentials.dvx))/(3*CONST.Q*nbarHMax)
    # print("UxHMax", UxHMax)
    # print("vr2vx2_ran2", vr2vx2_ran2.T)
    # print("THMax", THMax)
    # input()

    if compute_errors and debrief:
        print(prompt+'VbarM_error: '+sval(vbarM_error))
        print(prompt+'TWall Maxwellian: '+sval(TMax))
        print(prompt+'TWall Half Maxwellian: '+sval(THMax))

    #   Option to view inputted profiles

        #   Plotting - will maybe add later

    #   Starting back at line 429 from IDL code

    print("Satisfaction condition: ", truncate)

    KH_Common = Kinetic_H_Common() #Common block for Kinetic_H
    KH2_Common = Kinetic_H2_Common() #Common blocks for Kinetic_H2
        
    if oldrun:
        # checks if the previous run satisfies the required conditions 
        if debrief: 
            print(prompt, 'Maximum Normalized change in nH2: ', sval(nDelta_nH2))
        if debrief and pause: 
            input("Press any key to continue")
        if nDelta_nH2 > truncate: 
            # goto fH_fH2_iterate I think we will have to make fH_fH2_iterate a function 
            # since we wont be reading old runs right now I am going to leave this as is 
            pass
    else:
        #   Entry point for fH_fH2 iteration : iterates through solving fh and fh2 until they satisfy boltzmans equation
        nDelta_nH2 = truncate + 1
        while nDelta_nH2 > truncate: # Used goto statements in IDL; changed here to while loop
            if debrief: 
                print(prompt, 'Maximum Normalized change in nH2: ', sval(nDelta_nH2))
            if debrief and pause: 
                input()

            iter += 1
            if debrief:
                print(prompt+'fH/fH2 Iteration: '+sval(iter))
            nH2s = nH2

            # interpolate fH data onto H2 mesh: fH -> fHM
            do_warn = 5e-3
            fHM = interp_fvrvxx(fH, kh_mesh, kh2_mesh, do_warn=do_warn, debug=interp_debug) 
            # print("fHM", fHM.shape)

            # Compute fH2 using Kinetic_H2
            H2compute_errors = compute_errors and H2debrief # is this accurate, how can it be equal to both? - GG 2/15
            
            # print("fH2", fH2)
            # print("shape", fH2.shape)
            # input()
            kh2_results = kinetic_h2(
                    kh2_mesh, mu, vxiM, fh2BC, GammaxH2BC, NuLoss, fHM, SH2, fH2, nHP, THP, KH2_Common,\
                    truncate=truncate, max_gen=max_gen, compute_h_source=True, ni_correct=True,\
                    compute_errors=H2compute_errors, plot=H2plot,debug=H2debug,debrief=H2debrief,pause=H2pause)
            
            fH2, nHP, THP, nH2, GammaxH2, VxH2, pH2, TH2, qxH2, qxH2_total, Sloss, \
                QH2, RxH2, QH2_total, AlbedoH2, WallH2, fSH, SH, SP, SHP, NuE, NuDis, ESH, Eaxis, error = kh2_results

            # print("fH2", fH2.T)
            # print("nHP", nHP)
            # print("THP", THP)
            # print("nH2", nH2)
            # print("GammaxH2", GammaxH2)
            # print("TH2", TH2)
            # print("qxH2_total", qxH2_total)
            # print("AlbedoH2", AlbedoH2)
            # print("fSH", fSH.T)
            # print("SH", SH)
            # print("SP", SP)
            # input()
            

            # Interpolate H2 data onto H mesh: fH2 -> fH2A, fSH -> fSHA, nHP -> nHPA, THP -> THPA
            do_warn = 5.0E-3
            fH2A = interp_fvrvxx(fH2, kh2_mesh, kh_mesh, do_warn=do_warn, debug=interp_debug) 
            fSHA = interp_fvrvxx(fSH, kh2_mesh, kh_mesh, do_warn=do_warn, debug=interp_debug) #NOTE return value here not correct, see _Wxa calculation, set debug_flag
            # ii = np.nonzero(fH2A.reshape(fH2A.size, order='F'))
            # print("fH2Anz", ii)
            # print("fH2A", fH2A.reshape(fH2A.size, order='F')[ii])
            # ii = np.nonzero(fSHA.reshape(fSHA.size, order='F')) #NOTE Not Correct, Revisit
            # print("fSHAnz", ii)
            # print("fSHA", fSHA.reshape(fSHA.size, order='F')[ii])
            # input()

            nHPA = np.interp(kh_mesh.x, kh2_mesh.x, nHP, left=0, right=0)
            THPA = np.interp(kh_mesh.x, kh2_mesh.x, THP, left=0, right=0)
            # print("nHPA", nHPA)
            # print("THPA", THPA)
            # input()    

            # Compute fH using Kinetic_H
            GammaxHBC = 0
            # fHBC = np.zeros((kh_mesh.vr.size,kh_mesh.vx.size,kh_mesh.x.size))
            fHBC = np.zeros((kh_mesh.vr.size,kh_mesh.vx.size))
            ni_correct = 1
            Hcompute_errors = compute_errors and Hdebrief

            kh_results = kinetic_h(
                    kh_mesh, mu, vxiA, fHBC, GammaxHBC, fH2A, fSHA, fH, nHPA, THPA, jh_coefficients, KH_Common,
                    truncate=truncate, max_gen=max_gen, ni_correct=ni_correct, compute_errors=Hcompute_errors,
                    plot=Hplot, debug=Hdebug, debrief=Hdebrief, pause=Hpause) # Not sure where some of the keywords are defined
            
            fH,nH,GammaxH,VxH,pH,TH,qxH,qxH_total,NetHSource,Sion,QH,RxH,QH_total,AlbedoH,SideWallH,error = kh_results

            # print("fH", fH.T)
            # input()
            # print("nH", nH)
            # print("GammaxH2", GammaxH2)
            # print("TH", TH)
            # print("qxH_total", qxH_total)
            # print("NetHSource", NetHSource)
            # print("Sion", Sion)
            # print("QH_total", QH_total)
            # print("SideWallH", SideWallH)
            # input()


            # Interpolate SideWallH data onto H2 mesh: SideWallH -> SideWallHM
            SideWallHM = np.interp(kh2_mesh.x, kh_mesh.x, SideWallH, left=0, right=0)
            # print("SideWallHM", SideWallHM)
            # input()

            # Adjust SpH2 to achieve net zero hydrogen atom/molecule flux from wall
            # (See notes "Procedure to adjust the normalization of the molecular source at the 
            # limiters (SpH2) to attain a net zero atom/molecule flux from wall")

            # Compute SI, GammaH2Wall_minus, and GammaHWall_minus
            SI = np.trapezoid(SpH2, kh2_mesh.x)
            SwallI = np.trapezoid(0.5*SideWallHM, kh2_mesh.x)
            GammaH2Wall_minus = AlbedoH2*GammaxH2BC
            GammaHWall_minus = -GammaxH[0]
            # print("SI", SI)
            # print("SwallI", SwallI)
            # print("GammaH2Wall_minus", GammaH2Wall_minus)
            # print("GammaHWall_minus", GammaHWall_minus)
            # input()

            # Compute Epsilon and alphaplus1RH0Dis
            Epsilon = 2*GammaH2Wall_minus / (SI+SwallI)
            alphaplus1RH0Dis = GammaHWall_minus / ((1 - 0.5*Epsilon)*(SI + SwallI) + GammaxH2BC)
            # print("Epsilon", Epsilon)
            # print("alphaplus1RH0Dis", alphaplus1RH0Dis)
            # input()

            # Compute flux error, EH, and dEHdSI
            EH = 2*GammaxH2[0] - GammaHWall_minus
            dEHdSI = -Epsilon - alphaplus1RH0Dis*(1 - 0.5*Epsilon)
            # print("EH", EH)
            # print("dEHdSI", dEHdSI)
            # input()

            # Option: print normalized flux error
            nEH = np.abs(EH) / np.max(np.abs(np.array([2*GammaxH2[0], GammaHWall_minus] )))
            if debrief and compute_errors:
                print(prompt, 'Normalized Hydrogen Flux Error: ', sval(nEH))
            
            # Compute Adjustment 
            Delta_SI = -EH/dEHdSI
            SI = SI + Delta_SI
            # print("Delta_SI", GammaH2Wall_minus)
            # print("SI", GammaHWall_minus)
            # input()

            # Rescale SpH2 to have new integral value, SI
            SpH2 = SI*SpH2_hat
            EH_hist = np.append(EH_hist, EH)
            SI_hist = np.append(SI_hist, SI)
            # print("SpH2", SpH2)
            # print("EH_hist", EH_hist)
            # print("SI_hist", SI_hist)
            # input()

            # Set total H2 source
            SH2 = SpH2 + 0.5*SideWallHM
            # print("SH2", SH2)
            # input()

            # print("RxH_H2", KH2_Common.Output.RxH_H2)
            # print("RxH2_H", KH_Common.Output.RxH2_H)
            # input()
            if compute_errors:
                _RxH_H2 = np.interp(kh_mesh.x, kh2_mesh.x, KH2_Common.Output.RxH_H2, left=0, right=0)
                DRx = _RxH_H2 + KH_Common.Output.RxH2_H
                nDRx = np.max(np.abs(DRx)) / np.max(np.abs(np.array([_RxH_H2, KH_Common.Output.RxH2_H])))
                if debrief:
                    print(prompt, 'Normalized H2 <-> H Momentum Transfer Error: ', sval(nDRx))
            
            Delta_nH2 = np.abs(nH2 - nH2s)
            nDelta_nH2 = np.max(Delta_nH2/np.max(nH2))
    
    # fH_fH2_done code section  

    # NOTE Removed calculations for GammaHLim, gammaxh, gammaxH2, etc, Implement Later

    
    # Compute Lyman and Balmer NOTE These are not functioning yet
    Lyman = 0
    Balmer = 0
    # Lyman = lyman_alpha(kh_mesh.ne, kh_mesh.Te, nH, jh_coefficients, no_null = 1) #NOTE Not Working Yet
    # Balmer = balmer_alpha(kh_mesh.ne, kh_mesh.Te, nH, jh_coefficients, no_null = 1) #NOTE Not Working Yet


    # Store Outputs
    output_file = 'Results/output'
    print(prompt, "Saving files to", output_file+".npz")
    np.savez(output_file, xH2=kh2_mesh.x, nH2=nH2, GammaxH2=GammaxH2, TH2=TH2, qxH2_total=qxH2_total, nHP=nHP, THP=THP, SH=SH, SP=SP,
             xH=kh_mesh.x, nH=nH, GammaxH=GammaxH, TH=TH, qxH_total=qxH_total, NetHSource=NetHSource, Sion=Sion, QH_total=QH_total, SideWallH=SideWallH, Lyman=Lyman, Balmer=Balmer)
    

    # NOTE Add plotting later

    results = {}
    results["xH2"] = kh2_mesh.x
    results["nH2"] = nH2
    results["GammaxH2"] = GammaxH2
    results["TH2"] = TH2
    results["qxH2_total"] = qxH2_total
    results["nHP"] = nHP
    results["THP"] = THP
    results["SH"] = SH
    results["SP"] = SP
    results["xH"] = kh_mesh.x
    results["nH"] = nH
    results["GammaxH"] = GammaxH
    results["TH"] = TH
    results["qxH_total"] = qxH_total
    results["NetHSource"] = NetHSource
    results["Sion"] = Sion
    results["QH_total"] = QH_total
    results["SideWallH"] = SideWallH
    results["Lyman"] = Lyman
    results["Balmer"] = Balmer

    return results
