import numpy as np
import copy
from warnings import warn
from scipy import interpolate

from .make_dvr_dvx import VSpace_Differentials
from .locate import locate
from .utils import sval
from .kinetic_mesh import kinetic_mesh

from .common import constants as CONST

def interp_fvrvxx(fa, mesh1 : kinetic_mesh, mesh2 : kinetic_mesh, do_warn=None, debug=0, correct=1, debug_flag = 0): #NOTE Debug flag added to mark specific function calls, remove later
    
    #  Input:
    #     Input Distribution function 'a'
    #	fa	- dblarr(nVra,nVxa,nXa) distribution function
    #	Vra	- fltarr(nVra) - radial velocity
    #	Vxa	- fltarr(nVxa) - axial velocity
    #       Xa	- fltarr(nXa)  - spatial coordinate
    #       Tnorma	- float,  Normalization temperature for Vra & Vxa
    #
    #    Desired phase space coordinates of Output Distribution function 'b'
    #	Vrb	- fltarr(nVrb) - radial velocity
    #	Vxb	- fltarr(nVxb) - axial velocity
    #       Xb	- fltarr(nXb)  - spatial coordinate
    #       Tnormb	- float,  Normalization temperature for Vrb & Vxb
    #
    #  Output:
    #     Interpolated Distribution function 'b'
    #	fb	- dblarr(nVrb,nVxb,nXb) distribution function
    #	          fb is scaled if necessary to make its
    #	          digital integral over all velocity space
    #		  equal to that of fa.
    #
    #  Keywords:
    #     Input:
    #	do_warn	- float, acceptable truncation level.
    #		  For interpolations outside the phase space set by
    #		  (Vra, Vxa, Xa), the values of fb are set to zero.
    #		  This may not be acceptable. A test is performed on
    #		  fb at the boundaries. If fb at the boundaries is greater
    #		  than do_warn times the maximum value of fb,
    #		  a warning message is generated.

    prompt='INTERP_FVRVXX => '
    
    # Check shape agreement for fa
    if fa.shape == (mesh1.vr.size, mesh1.vx.size, mesh1.x.size):
        raise Exception('fa (' + str(fa.shape) + ') does not have shape (vra, vxa, xa)' + str((mesh1.vr.size, mesh1.vx.size, mesh1.x.size)))

    v_scale = np.sqrt(mesh2.Tnorm / mesh1.Tnorm) # velocity ratio (scales velocities from mesh_a to mesh_b)

    oki = np.where((v_scale*mesh2.vr <= max(mesh1.vr)) & (v_scale*mesh2.vr >= min(mesh1.vr)))[0]
    if len(oki) < 1:
        raise Exception('No values of Vrb are within range of Vra')
    i0, i1 = oki[0], oki[-1]

    okj = np.where((v_scale*mesh2.vx <= max(mesh1.vx)) & (v_scale*mesh2.vx >= min(mesh1.vx)))[0]
    if okj.size < 1:
        raise Exception('No values of Vxb are within range of Vxa')
    j0, j1 = okj[0], okj[-1]

    okk = np.where((mesh2.x <= max(mesh1.x)) & (mesh2.x >= min(mesh1.x)))[0]
    if okk.size < 1:
        raise Exception('No values of Xb are within range of Xa')
    k0, k1 = okk[0], okk[-1]

    nvrb = mesh2.vr.size
    nvxb = mesh2.vx.size
    nxb = mesh2.x.size

    fb = np.zeros((nvrb, nvxb, nxb))

    # --- Generate differentials ---
    #NOTE Move values into functions later
    differentials_a = VSpace_Differentials(mesh1.vr, mesh1.vx)
    Vr2pidVra = differentials_a.dvr_vol
    dVxa = differentials_a.dvx
    vraL = differentials_a.vr_left_bound
    vraR = differentials_a.vr_right_bound
    vxaL = differentials_a.vx_left_bound
    vxaR = differentials_a.vx_right_bound
    Vra2Vxa2 = differentials_a.vmag_squared

    differentials_b = VSpace_Differentials(mesh2.vr, mesh2.vx)
    Vr2pidVrb = differentials_b.dvr_vol
    dVxb = differentials_b.dvx
    vrbL = differentials_b.vr_left_bound
    vrbR = differentials_b.vr_right_bound
    vxbL = differentials_b.vx_left_bound
    vxbR = differentials_b.vx_right_bound
    Vol = differentials_b.volume
    Vth_DVx = differentials_b.vth_dvx
    Vx_DVx = differentials_b.vx_dvx
    Vr_DVr = differentials_b.vr_dvr
    Vrb2Vxb2 = differentials_b.vmag_squared
    jpa = differentials_b.vx_fpi
    jpb = differentials_b.vx_lpi
    jna = differentials_b.vx_fni
    jnb = differentials_b.vx_lni


    # NOTE Removed saving weights temporarily, re-implement later

    # --- Compute Weights ---

    #   Determine Left and Right limits on 'cells' for Vra, Vxa, Vrb, Vxb

    if debug:
        print(prompt+'computing new weight')

    #   Set area contributions to Weight array

    _weight = np.zeros((nvrb,nvxb,mesh1.vr.size,mesh1.vx.size))
    weight = np.zeros((nvrb*nvxb,mesh1.vr.size*mesh1.vx.size))

    for ib in range(nvrb):
        for jb in range(nvxb):
            for ia in range(mesh1.vr.size):
                vraMin = max([v_scale*vrbL[ib], vraL[ia]])
                vraMax = min([v_scale*vrbR[ib], vraR[ia]])
                for ja in range(mesh1.vx.size):
                    vxaMin = max([v_scale*vxbL[jb], vxaL[ja]])
                    vxaMax = min([v_scale*vxbR[jb], vxaR[ja]])
                    if vraMax > vraMin and vxaMax > vxaMin:
                        _weight[ib,jb,ia,ja] = 2*np.pi*(vraMax**2 - vraMin**2)*(vxaMax - vxaMin) / (Vr2pidVrb[ib]*dVxb[jb])

    weight = np.reshape(_weight, weight.shape, order = 'F') # previous version caused error

    # print("weight", weight.T[np.nonzero(weight.T)].size)
    # input()

    fb_xa = np.zeros((nvrb*nvxb,mesh1.x.size)) 

    #   Determine fb_xa from weight array

    _fa = np.reshape(fa, (mesh1.vr.size*mesh1.vx.size, mesh1.x.size), order = 'F')
    fb_xa = weight @ _fa

    # print("fb_xa", fb_xa)
    # print(fb_xa.shape)
    # input()

    #   Compute _Wxa and _Ea - these are the desired moments of fb, but on the xa grid

    na = np.zeros(mesh1.x.size)
    _Wxa = np.zeros(mesh1.x.size)
    _Ea = np.zeros(mesh1.x.size)

    for k in range(mesh1.x.size):
        na[k] = np.sum(Vr2pidVra*(fa[:,:,k] @ dVxa))
        if na[k] > 0:
            _Wxa[k] = np.sqrt(mesh1.Tnorm)*np.sum(Vr2pidVra*(fa[:,:,k] @ (mesh1.vx*dVxa)))/na[k]
            if debug_flag:
                print("_wxa", _Wxa[k])
                print("a", fa[:,:,k].T)
                print("b", (mesh1.vx*dVxa))
                print("c", (fa[:,:,k] @ (mesh1.vx*dVxa)))
                input()
            _Ea[k] = mesh1.Tnorm*np.sum(Vr2pidVra*((Vra2Vxa2*fa[:,:,k]) @ dVxa))/na[k]
    # print("na", na)
    # input()

    wxa = np.zeros(nxb)
    Ea = np.zeros(nxb)

    for k in range(k0, k1+1):
        kL = np.maximum(locate(mesh1.x, mesh2.x[k]), 0)
        kR = np.minimum(kL+1, mesh1.x.size-1)
        kL = np.minimum(kL, kR-1)

        f = (mesh2.x[k] - mesh1.x[kL]) / (mesh1.x[kR] - mesh1.x[kL])
        fb[:,:,k] = np.reshape((fb_xa[:,kL] + (fb_xa[:,kR] - fb_xa[:,kL])*f), fb[:,:,k].shape, order='F')
        wxa[k] = _Wxa[kL] + (_Wxa[kR] - _Wxa[kL])*f
        if debug_flag:
            print("klkr", kL, kR)
            print("wxa1", wxa[k])
            print("wxa2", _Wxa[kL])
            print("wxa3", _Wxa[kR])
            input()
        Ea[k]=_Ea[kL] + (_Ea[kR] - _Ea[kL])*f
    # print("fb", fb)
    # print("wxa", wxa)
    # print("Ea", Ea)
    # input()

    #   Correct fb so that it has the same Wx and E moments as fa

    if correct:

        #   Process each spatial location

        AN = np.zeros((nvrb, nvxb, 2))
        BN = np.zeros((nvrb, nvxb, 2))

        sgn = np.array([1, -1])

        for k in range(nxb):
            allow_neg = 0

            #   Compute nb, Wxb, and Eb - these are the current moments of fb

            nb = np.sum(Vr2pidVrb*(fb[:,:,k] @ dVxb))
            # print("nb", nb)
            # input()
            if nb > 0:
                
                #   Entry point for iteration - 'correct' tag in original code
                #   Since Python doesn't have goto, a while loop was used

                goto_correct = True
                while goto_correct:
                    goto_correct = False
                    nb = np.sum(Vr2pidVrb*(fb[:,:,k] @ dVxb))
                    Wxb = np.sqrt(mesh2.Tnorm)*np.sum(Vr2pidVrb*(fb[:,:,k] @ (mesh2.vx*dVxb))) / nb
                    Eb = mesh2.Tnorm*np.sum(Vr2pidVrb*((Vrb2Vxb2*fb[:,:,k]) @ dVxb)) / nb
                    # print("nb", nb)
                    # print("Wxb", Wxb)
                    # print("Eb", Eb)
                    # input()

                    #   Compute Nij from fb, padded with zeros

                    Nij = np.zeros((nvrb+2,nvxb+2))
                    Nij[1:nvrb+1,1:nvxb+1] = fb[:,:,k]*Vol / nb

                    #   Set Cutoff and remove Nij very close to zero

                    cutoff = 1.0e-6*np.max(Nij)
                    ii = np.where((abs(Nij) < cutoff) & (abs(Nij) > 0))
                    if ii[0].size > 0:
                        Nij[ii] = 0.0

                    if max(Nij[2,:]) <= 0:
                        allow_neg = 1

                    Nijp1_vx_Dvx = np.roll(Nij*Vx_DVx, shift=-1, axis=1)
                    Nij_vx_Dvx = Nij*Vx_DVx
                    Nijm1_vx_Dvx = np.roll(Nij*Vx_DVx, shift=1, axis=1)
                    Nip1j_vr_Dvr = np.roll(Nij*Vr_DVr, shift=-1, axis=0)
                    Nij_vr_Dvr = Nij*Vr_DVr
                    Nim1j_vr_Dvr = np.roll(Nij*Vr_DVr, shift=1, axis=0)

                    #   Compute Ap, Am, Bp, and Bm (0=p 1=m)

                    _AN                 = np.roll(Nij*Vth_DVx, shift=1, axis=1) - Nij*Vth_DVx
                    AN[:,:,0]           = copy.copy(_AN[1:nvrb+1,1:nvxb+1])
                    
                    _AN                 = -np.roll(Nij*Vth_DVx, shift=-1, axis=1) + Nij*Vth_DVx
                    AN[:,:,1]           = copy.copy(_AN[1:nvrb+1,1:nvxb+1])

                    BN[:,jpa+1:jpb+1,0] =  Nijm1_vx_Dvx[1:nvrb+1,jpa+2:jpb+2] - Nij_vx_Dvx[1:nvrb+1,jpa+2:jpb+2]
                    BN[:,jpa,0]         = -Nij_vx_Dvx[1:nvrb+1,jpa+1]
                    BN[:,jnb,0]         =  Nij_vx_Dvx[1:nvrb+1,jnb+1]
                    BN[:,jna:jnb,0]     = -Nijp1_vx_Dvx[1:nvrb+1,jna+1:jnb+1] + Nij_vx_Dvx[1:nvrb+1,jna+1:jnb+1]
                    BN[:,:,0]           =  BN[:,:,0] + Nim1j_vr_Dvr[1:nvrb+1,1:nvxb+1] - Nij_vr_Dvr[1:nvrb+1,1:nvxb+1]

                    BN[:,jpa+1:jpb+1,1] = -Nijp1_vx_Dvx[1:nvrb+1,jpa+2:jpb+2] + Nij_vx_Dvx[1:nvrb+1,jpa+2:jpb+2]
                    BN[:,jpa,1]         = -Nijp1_vx_Dvx[1:nvrb+1,jpa+1]
                    BN[:,jnb,1]         =  Nijm1_vx_Dvx[1:nvrb+1,jnb+1]
                    BN[:,jna:jnb,1]     =  Nijm1_vx_Dvx[1:nvrb+1,jna+1:jnb+1] - Nij_vx_Dvx[1:nvrb+1,jna+1:jnb+1]
                    BN[1:nvrb,:,1]      =  BN[1:nvrb,:,1] - Nip1j_vr_Dvr[2:nvrb+1,1:nvxb+1] + Nij_vr_Dvr[2:nvrb+1,1:nvxb+1]
                    BN[0,:,1]           =  BN[0,:,1] - Nip1j_vr_Dvr[1,1:nvxb+1]

                    # print("AN", AN.T)
                    # print("BN", BN.T)
                    # input()

                    #   If negative values for Nij must be allowed, then add postive particles to i=0 and negative particles to i=1 (beta is negative here)

                    if allow_neg:
                        BN[0,:,1] = BN[0,:,1] - Nij_vr_Dvr[1,1:nvxb+1]
                        BN[1,:,1] = BN[1,:,1] + Nij_vr_Dvr[1,1:nvxb+1]

                    #   Remove padded zeros in Nij

                    Nij = Nij[1:nvrb+1,1:nvxb+1]
                    

                    #   Cycle through 4 possibilies of sign(alpha),sign(beta)

                    TB1 = np.zeros(2, float)
                    TB2 = np.zeros(2, float)

                    for ia in range(2):
                        # print("a")
                        #   Compute TA1, TA2

                        TA1 = np.sqrt(mesh2.Tnorm)*np.sum((AN[:,:,ia] @ mesh2.vx))
                        TA2 = mesh2.Tnorm*np.sum(Vrb2Vxb2*AN[:,:,ia])
                        for ib in range(2):
                            # print("b")

                            #   Compute TB1, TB2

                            if TB1[ib] == 0:
                                TB1[ib] = np.sqrt(mesh2.Tnorm)*np.sum((BN[:,:,ib] @ mesh2.vx))
                            if TB2[ib] == 0:
                                TB2[ib] = mesh2.Tnorm*np.sum(Vrb2Vxb2*BN[:,:,ib])

                            denom = TA2*TB1[ib] - TA1*TB2[ib]
                            if debug_flag:
                                print("denom", denom)
                                # print(TA2*TB1[ib])
                                # print(TA1*TB2[ib])
                                # print(TB2[ib])
                                # print(BN[:,:,ib].T)
                                input()
                            beta = 0
                            alpha = 0

                            if (denom != 0) and (TA1 != 0):
                                beta = (TA2*(wxa[k] - Wxb) - TA1*(Ea[k] - Eb))/denom # fixed capitalization
                                alpha = (wxa[k] - Wxb - TB1[ib]*beta)/TA1

                            do_break = ((alpha*sgn[ia]) > 0) and ((beta*sgn[ib]) > 0)
                            # print("break", do_break)
                            # print(Wxb)
                            if do_break:
                                break
                        if do_break:
                            break

                    #   Entry point for 'alpha_beta' tag from original code

                    RHS = AN[:,:,ia]*alpha + BN[:,:,ib]*beta

                    #   Are there locations where Nij = 0.0 and RHS is negative?
                    #   ii=where(Nij eq 0.0 and RHS lt 0.0,count) was in the original code, I don't think it does anything

                    s = 1
                    if not allow_neg:
                        ii = np.nonzero(Nij)
                        if np.size(ii) > 0:
                            s = min(1/np.max(-RHS[ii]/Nij[ii]),1)
                    
                    fb[:,:,k] = nb*(Nij + s*RHS)/Vol # fixed capitalization

                    # print("goto", goto_correct)
                    goto_correct = (s < 1)
    if(debug_flag != 0):
        ii = np.nonzero(fb.reshape(fb.size, order='F'))
        print("fSHAnz", ii)
        input()

    if do_warn != None:

        #   Test Boundaries:

        #   i0 & i1 

        big = np.max(fb)

        i0_error = 0
        i1_error = 0
        if (i0 > 0) or (i1 < nvrb-1):
            for k in range(k0, k1+1):
                for j in range(j0, j1+1):
                    if (i0_error == 0) and (i0 > 0) and (fb[i0,j,k] > do_warn*big):
                        warn('Non-zero value of fb detected at min(Vra) boundary')
                        i0_error = 1
                    if (i1_error == 0) and (i1 < nvrb-1) and (fb[i1,j,k] > do_warn*big): # fixed capitalization
                        warn('Non-zero value of fb detected at max(Vra) boundary')
                        i1_error = 1

        #   j0 & j1

        j0_error = 0
        j1_error = 0
        if (j0 > 0) or (j1 < nvxb-1):
            for k in range(k0, k1+1):
                for i in range(i0, i1+1):
                    if (j0_error == 0) and (j0 > 0) and (fb[i,j0,k] > do_warn*big):
                        warn('Non-zero value of fb detected at min(Vxa) boundary')
                        j0_error=1
                    if (j1_error == 0) and (j1 < nvxb-1) and (fb[i,j1,k] > do_warn*big): # fixed capitalization
                        warn('Non-zero value of fb detected at max(Vxa) boundary')
                        j1_error=1

        #   k0 & k1

        k0_error = 0
        k1_error = 0
        if (k0 > 0) or (k1 < nxb-1):
            for i in range(i0, i1+1):
                for j in range(j0, j1+1):
                    if (k0_error == 0) and (k0 > 0) and (fb[i,j,k0] > do_warn*big):
                        warn('Non-zero value of fb detected at min(Xa) boundary')
                        k0_error = 1
                    if (k1_error == 0) and (k1 < nxb-1) and (fb[i,j,k1] > do_warn*big):
                        warn('Non-zero value of fb detected at max(Xa) boundary')
                        k1_error = 1

    #   Rescale

    tot_a = np.zeros(mesh1.x.size)
    for k in range(mesh1.x.size):
        tot_a[k] = np.sum(Vr2pidVra*(fa[:,:,k] @ dVxa))
    tot_b = np.zeros(nxb)
    tot_b[k0:k1+1] = interpolate.interp1d(mesh1.x,tot_a,fill_value="extrapolate")(mesh2.x[k0:k1+1])
    ii = np.where(fb>0)
    if ii[0].size > 0: # replaced fb with ii
        min_tot = np.min(np.array(fb[ii])) #(np.array([fb[tuple(i)] for i in ii]))
        for k in range(k0, k1+1):
            tot = np.sum(Vr2pidVrb*(fb[:,:,k] @ dVxb))
            if tot > min_tot:
                if debug:
                    print(prompt+'Density renormalization factor ='+sval(tot_b[k]/tot))
                fb[:,:,k] = fb[:,:,k]*tot_b[k]/tot

    if debug:

        #   Compute Vtha, Vtha2, Vthb and Vthb2
        mass = 1*CONST.H_MASS # mu*hydrogen mass
        vth1 = np.sqrt((2*CONST.Q*mesh1.Tnorm) / mass)
        vth2 = np.sqrt((2*CONST.Q*mesh2.Tnorm) / mass)

        #   na, Uxa, Ta
        na = np.zeros(mesh1.x.size)
        Uxa = np.zeros(mesh1.x.size)
        Ta = np.zeros(mesh1.x.size)
        vr2vx2_ran2 = np.zeros((mesh1.vr.size,mesh1.vx.size)) # fixed np.zeros() call

        for k in range(mesh1.x.size):
            na[k] = np.sum(Vr2pidVra*(fa[:,:,k] @ dVxa)) # fixed capitalization
            if na[k] > 0:
                Uxa[k] = vth1*np.sum(Vr2pidVra*(fa[k,:,:] @ (mesh1.vx*dVxa)))/na[k]
                for i in range(mesh1.vr.size):
                    vr2vx2_ran2[i,:] = mesh1.vr[i]**2 + (mesh1.vx - Uxa[k]/vth1)**2 # fixed capitalization
                Ta[k] = mass*(vth1**2)*np.sum(Vr2pidVra*((vr2vx2_ran2*fa[:,:,k]) @ dVxa))/(3*CONST.Q*na[k])

        #   nb, Uxb, Tb
        nb = np.zeros(nxb)
        Uxb = np.zeros(nxb)
        Tb = np.zeros(nxb)
        vr2vx2_ran2 = np.zeros((nvrb,nvxb)) # fixed np.zeros() call

        for k in range(nxb):
            nb[k] = np.sum(Vr2pidVrb*(fb[:,:,k] @ dVxb))
            if nb[k] > 0:
                Uxb[k] = vth2*np.sum(Vr2pidVrb*(fb[:,:,k] @ (mesh2.vx*dVxb)))/nb[k] # fixed typo
                for i in range(nvrb):
                    vr2vx2_ran2[i,:] = mesh2.vr[i]**2 + (mesh2.vx - Uxb[k]/vth2)**2 # fixed capitalization
                Tb[k] = mass*(vth2**2)*np.sum(Vr2pidVrb*((vr2vx2_ran2*fb[:,:,k]) @ dVxb))/(3*CONST.Q*nb[k])

        #   Plotting stuff was here in the original code
        #   May be added later, but has been left out for now

    return fb
