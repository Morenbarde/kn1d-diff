import numpy as np
import torch

from .make_dvr_dvx import VSpace_Differentials
from .common import constants as CONST

def compensate_distribution(f_slice, vdiff, vr, vx, vth, target_vx, target_energy, nb = 1, assume_pos = True):
    '''
    Custom Compensation scheme to give a distribution desired moments. Performs on one slice of the distribution.

    Parameters
    ----------
        f_slice : ndarray
            Slice of distribution fucntion
        vdiff : VSpace_Differentials
            Velocity space differentials for distribution function
        vr : ndarray
            Radial Velocities
        vx : ndarray
            Axial Velocities
        vth : ndarray
            Thermal Velocities
        target_vx : ndarray
            Target vx moment for distribution slice
        target_energy : ndarray
            Target energy moment for distribution slice
        nb : ndarray
            Density factor (Used in interp_fvrvxx)
        assume_pos : bool
            Sets whether the function assumes the distribution is positive
            Defaults to (True)

    Returns
    -------
        f_slice : ndarray
            Adjusted distribution function
        s : float
            Correction scalar (Used in interp_fvrvxx)
    '''
    
    #NOTE Get nb name checked

    nvr = vr.numel()
    nvx = vx.numel()


    #Differential Values torch conversion, replace later with make_dvr_dvx conversion
    dvr_vol = torch.from_numpy(vdiff.dvr_vol)
    dvx = torch.from_numpy(vdiff.dvx)
    vmag_squared = torch.from_numpy(vdiff.vmag_squared)
    volume = torch.from_numpy(vdiff.volume)


    # vth_diffs = np.zeros((vr.size, vx.size, 2), float)
    vth_diffs = torch.zeros((nvr, nvx, 2))
    # vrvx_diffs = np.zeros((vr.size, vx.size, 2), float)
    vrvx_diffs = torch.zeros((nvr, nvx, 2))

    # Compute present moments of Maxwell, WxMax, and EMax (x_moment, energy_moment)
    # vx_moment = vth*np.sum(dvr_vol*np.matmul(f_slice, (vx*dvx))) / nb
    # energy_moment = (vth**2)*np.sum(dvr_vol*np.matmul((vmag_squared*f_slice), dvx)) / nb
    vx_moment = vth*torch.sum(dvr_vol*(f_slice @ (vx*dvx))) / nb
    energy_moment = (vth**2)*torch.sum(dvr_vol*((vmag_squared*f_slice) @ dvx)) / nb

    # Compute weighted function from distribution, padded with zeros
    # weighted_dist = np.zeros((vr.size+2, vx.size+2), dtype=np.float64) #NOTE Check with someone if this name is accurate
    weighted_dist = torch.zeros((nvr+2, nvx+2)) #NOTE Check with someone if this name is accurate

    #Shorthand slices for center of padded distribution
    vr_center = slice(1, nvr+1)
    vx_center = slice(1, nvx+1)

    weighted_dist[vr_center, vx_center] = f_slice*volume / nb

    # Used for interp_fvrvxx
    # Run additional correction if the distribution is not assumed to be positive
    allow_neg = False
    '''
    # NOTE Not Autodifferentaible yet, revisit later
    if not assume_pos:
        cutoff = 1.0e-6*np.max(weighted_dist)
        ii = np.where((abs(weighted_dist) < cutoff) & (abs(weighted_dist) > 0))
        if ii[0].size > 0:
            weighted_dist[ii] = 0.0

        if max(weighted_dist[2,:]) <= 0:
            allow_neg = True
    '''

    vx_dist = weighted_dist*torch.from_numpy(vdiff.vx_dvx)
    vr_dist = weighted_dist*torch.from_numpy(vdiff.vr_dvr)
    vth_dist = weighted_dist*torch.from_numpy(vdiff.vth_dvx)

    # Compute Ap, Am, Bp, and Bm (0=p 1=m)

    # diff_padded = np.roll(vth_dist, shift=1, axis=1) - vth_dist
    # vth_diffs[:,:,0]   = np.copy(diff_padded[vr_center, vx_center])
    diff_padded = torch.roll(vth_dist, shifts=1, dims=1) - vth_dist
    vth_diffs[:,:,0]   = torch.clone(diff_padded[vr_center, vx_center])
    
    # diff_padded = -np.roll(vth_dist, shift=-1, axis=1) + vth_dist
    # vth_diffs[:,:,1]   = np.copy(diff_padded[vr_center, vx_center])
    diff_padded = -torch.roll(vth_dist, shifts=-1, dims=1) + vth_dist
    vth_diffs[:,:,1]   = torch.clone(diff_padded[vr_center, vx_center])

    # Indices for pos/neg vx values
    pos_start = vdiff.vx_pos_start  # First positive vx index
    pos_end   = vdiff.vx_pos_end    # Last positive vx index
    neg_start = vdiff.vx_neg_start  # First negative vx index
    neg_end   = vdiff.vx_neg_end    # Last negative vx index

    vrvx_diffs[:, pos_start+1:pos_end+1, 0] =  vx_dist[vr_center, pos_start+1:pos_end+1]             - vx_dist[vr_center, pos_start+2:pos_end+2]
    vrvx_diffs[:, pos_start, 0]             = -vx_dist[vr_center, pos_start+1]
    vrvx_diffs[:, neg_end, 0]               =  vx_dist[vr_center, neg_end+1]
    vrvx_diffs[:, neg_start:neg_end, 0]     = -vx_dist[vr_center, neg_start+2:neg_end+2]             + vx_dist[vr_center, neg_start+1:neg_end+1]
    # vrvx_diffs[:,:,0]                      +=  vr_dist[0:vr.size, vx_center]             - vr_dist[vr_center, vx_center     ]
    vrvx_diffs[:,:,0]                       =  vrvx_diffs[:,:,0] + vr_dist[0:nvr, vx_center]         - vr_dist[vr_center, vx_center     ]

    vrvx_diffs[:, pos_start+1:pos_end+1, 1] = -vx_dist[vr_center, pos_start+3:pos_end+3]             + vx_dist[vr_center, pos_start+2:pos_end+2]
    vrvx_diffs[:, pos_start, 1]             = -vx_dist[vr_center, pos_start+2]
    vrvx_diffs[:, neg_end, 1]               =  vx_dist[vr_center, neg_end]
    vrvx_diffs[:, neg_start:neg_end, 1]     =  vx_dist[vr_center, neg_start:neg_end]                 - vx_dist[vr_center, neg_start+1:neg_end+1]
    # vrvx_diffs[1:vr.size, :, 1]            +=  vr_dist[2:vr.size+1, vx_center]           - vr_dist[3:vr.size+2, vx_center]
    vrvx_diffs[1:nvr, :, 1]                 =  vrvx_diffs[1:nvr, :, 1] + vr_dist[2:nvr+1, vx_center] - vr_dist[3:nvr+2, vx_center]
    # vrvx_diffs[0,:,1]                      -=  vr_dist[2, vx_center]
    vrvx_diffs[0,:,1]                       =  vrvx_diffs[0,:,1] - vr_dist[2, vx_center]

    #   If negative values for weighted distribution must be allowed, then add postive particles to i=0 and negative particles to i=1 (beta is negative here)
    if allow_neg:
        vrvx_diffs[0,:,1] = vrvx_diffs[0,:,1] - vr_dist[1,vx_center]
        vrvx_diffs[1,:,1] = vrvx_diffs[1,:,1] + vr_dist[1,vx_center]

    # Remove padded zeros in weighted distribution
    weighted_dist = weighted_dist[vr_center, vx_center]

    # Cycle through 4 possibilies of sign(a_Max),sign(b_Max)
    # TB1 = np.zeros(2, float)
    # TB2 = np.zeros(2, float)
    TB1 = torch.zeros(2)
    TB2 = torch.zeros(2)
    sign = [1,-1]
    
    for ia in range(2):

        # Compute TA1, TA2
        # TA1 = vth*np.sum(np.matmul(vth_diffs[:,:,ia], vx))
        # TA2 = (vth**2)*np.sum(vmag_squared*vth_diffs[:,:,ia])
        TA1 = vth*torch.sum(vth_diffs[:,:,ia] @ vx)
        TA2 = (vth**2)*torch.sum(vmag_squared*vth_diffs[:,:,ia])

        for ib in range(2):

            # Compute TB1, TB2
            if TB1[ib] == 0:
                # TB1[ib] = vth*np.sum(np.matmul(vrvx_diffs[:,:,ib], vx))
                TB1[ib] = vth*torch.sum(vrvx_diffs[:,:,ib] @ vx)

            if TB2[ib] == 0:
                # TB2[ib] = (vth**2)*np.sum(vmag_squared*vrvx_diffs[:,:,ib])
                TB2[ib] = (vth**2)*torch.sum(vmag_squared*vrvx_diffs[:,:,ib])

            denom = TA2*TB1[ib] - TA1*TB2[ib]

            vrvx_scalar = 0
            vth_scalar = 0
            if (denom != 0) and (TA1 != 0):
                vrvx_scalar = (TA2*(target_vx - vx_moment) - TA1*(target_energy - energy_moment)) / denom
                vth_scalar = (target_vx - vx_moment - TB1[ib]*vrvx_scalar) / TA1
                
        #     do_break = (vth_scalar*sign[ia] > 0) and (vrvx_scalar*sign[ib] > 0)
        #         # End While Loops
        #     if do_break:
        #         break
        # if do_break:
        #     break

    correction = vth_diffs[:,:,ia]*vth_scalar + vrvx_diffs[:,:,ib]*vrvx_scalar

    # Additional stuff for interp_fvrvxx
    s = 1
    '''
    # NOTE Not Autodifferentaible yet, revisit later
    if (not assume_pos) and (not allow_neg):
        ii = np.nonzero(weighted_dist)
        if np.size(ii) > 0:
            s = min(1/np.max(-correction[ii]/weighted_dist[ii]),1)
    '''

    f_slice = nb*(weighted_dist + s*correction) / volume

    return f_slice, s


# NOTE Look at PlasmaPy, specifically plasmapy.formulary.distribution.Maxwellian_velocity_2D
# NOTE Adjust Docstring for accuracy
def create_shifted_maxwellian(vr, vx, Tmaxwell, vx_shift, mu, mol, Tnorm):
    # NOTE
    # Need to differentiate Tmaxwell, vx_shift,
    # Do NOT Need to differentiate mu, mol, Tnorm
    # Maybe need to differentiate vr, vx if we decide to autodiff the grid
    # Sometimes autodiffing this is necessary, sometimes not, maybe add flag for torch.nograd?
    '''
    Creates a shifted maxwellian distrubution with a desired temperature and vx moment

    Parameters
    ----------
        vr : ndarray
            radial velocities
        vx : ndarray
            axial velocities
        Tmaxwell : ndarray
            Desired temperature moments, (eV)
        vx_shift : ndarray
            Desired vx moments, (m s^-1)
        mu : int
            1=hydrogen, 2=deuterium
        mol : int
            1=atom, 2=diatomic molecule
        Tnorm : ndarray
            Average Temperatures
            
    Returns
    -------
        Maxwell : ndarray
           3D array of shape (len(vr), len(vx), len(vx_shift)) 
           Shifted Maxwellian distribution function having numerically 
           evaluated vx moment close to Vx_shift and temperature close to Tmaxwell

    Notes
    -------
        One might think that Maxwell could be simply computed by a direct evaluation of the EXP function:

            for i=0,nvr-1 do begin
                arg=-(vr(i)^2+(vx-Vx_shift/vth)^2) * mol*Tnorm/Tmaxwell
                Maxwell(i,*,k)=exp(arg > (-80))
            endfor

        But owing to the discrete velocity space bins, this method does not necessarily lead to a digital representation 
        of a shifted Maxwellian (Maxwell) that when integrated numerically has the desired vx moment of Vx_shift
        and temperature, Tmaxwell.

        In order to insure that Maxwell has the desired vx and T moments when evaluated numerically, a compensation
        scheme is employed - similar to that used in Interp_fVrVxX. See compensate_distribution()
    '''

    dtype = Tmaxwell.dtype
    device = Tmaxwell.device

    # maxwell = np.zeros((vr.size, vx.size, vx_shift.size), float)
    vth = np.sqrt(2*CONST.Q*Tnorm / (mu*CONST.H_MASS)) #Thermal Velocity

    #--- Generate Velocity Differentials ---
    vdiff = VSpace_Differentials(vr.cpu().numpy(), vx.cpu().numpy())
    dvx = torch.from_numpy(vdiff.dvx)
    dvr_vol = torch.from_numpy(vdiff.dvr_vol)


    nvr = vr.numel()
    nvx = vx.numel()
    nk = vx_shift.numel()

    maxwell = torch.zeros((nvr, nvx, nk), dtype=dtype, device=device)

    for k in range(nk):
        if Tmaxwell[k] <= 0:
            continue

        arg = -((vr[:, None]**2 + (vx - (vx_shift[k] / vth))**2)*mol*Tnorm) / Tmaxwell[k]
        arg = torch.clamp(arg, min=-80.0, max=0.0)
        f = torch.exp(arg)

        f = f / torch.nansum(dvr_vol*(f @ dvx))

        # Target energy density
        target_energy = (vx_shift[k]**2) + (3*CONST.Q*Tmaxwell[k] / (mol*mu*CONST.H_MASS))

        with torch.no_grad():
            compensated_f, _ = compensate_distribution(f, vdiff, vr, vx, vth, vx_shift[k], target_energy)

        f = f + (compensated_f - f).detach()

        # f, _ = compensate_distribution(f, vdiff, vr, vx, vth, vx_shift[k], target_energy)

        maxwell[:,:,k] = f / torch.sum(dvr_vol*(f @ dvx))

    return maxwell
