import numpy as np

from .make_dvr_dvx import VSpace_Differentials
from .common import constants as CONST
from .utils import sval

def create_shifted_maxwellian(vr,vx,Tmaxwell,vx_shift,mu,mol,Tnorm):

    # NOTE Look at PlasmaPy, specifically plasmapy.formulary.distribution.Maxwellian_velocity_2D

    #   Input:
	#       Vx_shift  - dblarr(nx), (m s^-1)
    #       Tmaxwell  - dblarr(nx), (eV)
    #       Shifted_Maxwellian_Debug - if set, then print debugging information
    #       mol       - 1=atom, 2=diatomic molecule
 
    #   Output:
	#       Maxwell   - dblarr(nvr,nvx,nx) a shifted Maxwellian distribution function
	#	        having numerically evaluated vx moment close to Vx_shift and
	#	        temperature close to Tmaxwell

    #   Notes on Algorithm:

    #   One might think that Maxwell could be simply computed by a direct evaluation of the EXP function:

    #       for i=0,nvr-1 do begin
    #           arg=-(vr(i)^2+(vx-Vx_shift/vth)^2) * mol*Tnorm/Tmaxwell
    #           Maxwell(i,*,k)=exp(arg > (-80))
    #       endfor

    #   But owing to the discrete velocity space bins, this method does not necessarily lead to a digital representation 
    #   of a shifted Maxwellian (Maxwell) that when integrated numerically has the desired vx moment of Vx_shift
    #   and temperature, Tmaxwell.

    #   In order to insure that Maxwell has the desired vx and T moments when evaluated numerically, a compensation
    #   scheme is employed - similar to that used in Interp_fVrVxX

    shifted_maxwellian_debug = 0 #NOTE Move from here in future

    maxwell = np.zeros((vr.size, vx.size, vx_shift.size), float)
    vr2vx2_ran2 = np.zeros((vr.size, vx.size), float)
    vth = np.sqrt(2*CONST.Q*Tnorm / (mu*CONST.H_MASS)) #Thermal Velocity
    vth_squared = vth**2

    #Generate Velocity Differentials
    d_vspace = VSpace_Differentials(vr, vx)
    

    #NOTE get these names checked by a physicist
    vth_diffs = np.zeros((vr.size, vx.size, 2), float)
    vrvx_diffs = np.zeros((vr.size, vx.size, 2), float)
    sign = [1,-1]
    for k in range(vx_shift.size):
        if Tmaxwell[k] <= 0:
            break

        arg = -((vr[:, np.newaxis]**2 + (vx - (vx_shift[k] / vth))**2)*mol*Tnorm / Tmaxwell[k])
        arg = np.where(np.logical_and((-80 < arg), (arg < 0.0)), arg, -80)
        maxwell[:,:,k] = np.exp(arg)

        variable = np.matmul(maxwell[:,:,k], d_vspace.dvx)
        maxwell[:,:,k] = maxwell[:,:,k] / np.nansum(d_vspace.dvr_vol*variable)
        
        if shifted_maxwellian_debug:
            vx_out1 = vth*np.sum(d_vspace.dvr_vol*np.matmul((vx*d_vspace.dvx), maxwell[k,:,:]))
            for i in range(vr.size):
                vr2vx2_ran2[:,i] = vr[i]**2 + (vx - (vx_out1/vth))**2
            T_out1 = (mol*mu*CONST.H_MASS)*vth_squared*np.sum(d_vspace.dvr_vol*(np.matmul(d_vspace.dvx, vr2vx2_ran2*maxwell[k, :, :]))) / (3*CONST.Q)
            vth_local = 0.1*np.sqrt(2*Tmaxwell[k]*CONST.Q / (mol*mu*CONST.H_MASS))
            Terror = abs(Tmaxwell[k] - T_out1) / Tmaxwell[k]
            Verror = abs(vx_out1 - vx_shift[k]) / vth_local

        # Compute desired moments

        # NOTE get this name checked by a physicist
        # Target energy density
        target_energy = (vx_shift[k]**2) + (3*CONST.Q*Tmaxwell[k] / (mol*mu*CONST.H_MASS))

        # Compute present moments of Maxwell, WxMax, and EMax (x_moment, energy_moment)
        # NOTE get these names checked by a physicist
        vx_moment = vth*np.nansum(d_vspace.dvr_vol*np.dot(maxwell[:, :, k], (vx*d_vspace.dvx)))
        energy_moment = vth_squared*np.nansum(d_vspace.dvr_vol*np.dot((d_vspace.v_squared*maxwell[:, :, k]), d_vspace.dvx))

        # Compute Nij from Maxwell, padded with zeros
        weighted_maxwell = np.zeros((vr.size+2, vx.size+2), dtype=np.float64) #NOTE Check with someone if this name is accurate
        vr_slice = slice(1, vr.size+1)
        vr_slice_min1 = slice(0, vr.size)
        vx_slice = slice(1, vx.size+1)

        weighted_maxwell[vr_slice, vx_slice] = maxwell[:,:,k]*d_vspace.volume
        vx_maxwell = weighted_maxwell*d_vspace.vx_dvx
        vr_maxwell = weighted_maxwell*d_vspace.vr_dvr

        # Compute Ap, Am, Bp, and Bm (0=p 1=m)
        vth_maxwell = weighted_maxwell*d_vspace.vth_dvx

        diff_padded = np.roll(vth_maxwell, shift=1, axis=1) - vth_maxwell
        vth_diffs[:,:,0]   = np.copy(diff_padded[vr_slice, vx_slice])
        
        diff_padded = -np.roll(vth_maxwell, shift=-1, axis=1) + vth_maxwell
        vth_diffs[:,:,1]   = np.copy(diff_padded[vr_slice, vx_slice])

        # Define shorthand array slices
        pos_slice_plus1 = slice(d_vspace.pos_vx0+1, d_vspace.pos_vxn+1)  #NOTE Check for simplification later
        pos_slice_plus2 = slice(d_vspace.pos_vx0+2, d_vspace.pos_vxn+2)
        pos_slice_plus3 = slice(d_vspace.pos_vx0+3, d_vspace.pos_vxn+3)
        neg_slice       = slice(d_vspace.neg_vx0  , d_vspace.neg_vxn)
        neg_slice_plus1 = slice(d_vspace.neg_vx0+1, d_vspace.neg_vxn+1)
        neg_slice_plus2 = slice(d_vspace.neg_vx0+2, d_vspace.neg_vxn+2)

        vrvx_diffs[:, pos_slice_plus1, 0]       =  vx_maxwell[vr_slice, pos_slice_plus1]                    - vx_maxwell[vr_slice,pos_slice_plus2]
        vrvx_diffs[:, d_vspace.pos_vx0, 0]      = -vx_maxwell[vr_slice, d_vspace.pos_vx0+1]
        vrvx_diffs[:, d_vspace.neg_vxn, 0]      =  vx_maxwell[vr_slice, d_vspace.neg_vxn+1]
        vrvx_diffs[:, neg_slice, 0]             = -vx_maxwell[vr_slice, neg_slice_plus2]                    + vx_maxwell[vr_slice,neg_slice_plus1]
        vrvx_diffs[:,:,0]                      +=  vr_maxwell[vr_slice_min1, vx_slice]  #NOTE Can probably recombine these, but give slightly different result
        vrvx_diffs[:,:,0]                      -=  vr_maxwell[vr_slice, vx_slice]

        vrvx_diffs[:, pos_slice_plus1, 1]       = -vx_maxwell[vr_slice, pos_slice_plus3]                    + vx_maxwell[vr_slice,pos_slice_plus2]
        vrvx_diffs[:, d_vspace.pos_vx0, 1]      = -vx_maxwell[vr_slice, d_vspace.pos_vx0+2]
        vrvx_diffs[:, d_vspace.neg_vxn, 1]      =  vx_maxwell[vr_slice, d_vspace.neg_vxn]
        vrvx_diffs[:, neg_slice, 1]             =  vx_maxwell[vr_slice, neg_slice]                          - vx_maxwell[vr_slice, neg_slice_plus1]
        vrvx_diffs[1:vr.size, :, 1]            -=  vr_maxwell[3:vr.size+2, vx_slice]
        vrvx_diffs[1:vr.size, :, 1]            +=  vr_maxwell[2:vr.size+1, vx_slice]
        vrvx_diffs[0,:,1]                      -=  vr_maxwell[2, vx_slice]

        # Remove padded zeros in Nij
        weighted_maxwell = weighted_maxwell[vr_slice,vx_slice]

        # Cycle through 4 possibilies of sign(a_Max),sign(b_Max)
        # NOTE Make better names, discuss with physicist
        TB1 = np.zeros(2, float)
        TB2 = np.zeros(2, float)
        
        ia = 0
        while (ia < 2):

            # Compute TA1, TA2
            TA1 = vth*np.sum(np.matmul(vth_diffs[:,:,ia], vx))
            TA2 = vth_squared*np.sum(d_vspace.v_squared*vth_diffs[:,:,ia])

            ib = 0
            while (ib < 2):

                # Compute TB1, TB2
                if TB1[ib] == 0:
                    TB1[ib] = vth*np.sum(np.dot(vrvx_diffs[:,:,ib], vx))

                if TB2[ib] == 0:
                    TB2[ib] = vth_squared*np.sum(d_vspace.v_squared*vrvx_diffs[:,:,ib])

                denom = TA2*TB1[ib] - TA1*TB2[ib]

                # NOTE Check these names with physicists
                vrvx_scalar = 0
                vth_scalar = 0
                if (denom != 0) and (TA1 != 0):
                    vrvx_scalar = (TA2*(vx_shift[k] - vx_moment) - TA1*(target_energy - energy_moment)) / denom
                    vth_scalar = (vx_shift[k] - vx_moment - TB1[ib]*vrvx_scalar) / TA1
                    
                    # NOTE Some of these values are still off, but maxwell seems to be working for now
                if (vth_scalar*sign[ia] > 0) and (vrvx_scalar*sign[ib] > 0):
                    maxwell[:,:,k] = (weighted_maxwell + vth_diffs[:,:,ia]*vth_scalar + vrvx_diffs[:,:,ib]*vrvx_scalar) / d_vspace.volume
                    #End While Loops
                    ia = 2
                    ib = 2
                ib += 1
            ia += 1

        maxwell[:,:,k] /= np.sum(d_vspace.dvr_vol*(np.matmul(maxwell[:, :, k], d_vspace.dvx)))

        if shifted_maxwellian_debug:
            vx_out2 = vth*np.sum(d_vspace.dvr_vol*np.matmul((vx*d_vspace.dvx), maxwell[k,:,:]))
            for i in range(vr.size):
                vr2vx2_ran2[:,i] = vr[i]**2 + (vx - (vx_out2/vth))**2
            T_out2 = (mol*mu*CONST.H_MASS)*vth_squared*np.sum(d_vspace.dvr_vol*np.matmul(d_vspace.dvx, vr2vx2_ran2*maxwell[k,:,:])) / (3*CONST.Q)
            Terror2 = abs(Tmaxwell[k] - T_out2) / Tmaxwell[k]
            Verror2 = abs(vx_shift[k] - vx_out2) / vth_local
            print('CREATE_SHIFTED_MAXWELLIAN=> Terror:' + sval(Terror) + '->' + sval(Terror2) + '  Verror:' + sval(Verror)+'->' + sval(Verror2))

    return maxwell
