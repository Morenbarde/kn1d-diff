import numpy as np
from warnings import warn
from scipy import interpolate

from .make_dvr_dvx import VSpace_Differentials
from .create_shifted_maxwellian import compensate_distribution
from .locate import locate
from .utils import sval
from .kinetic_mesh import kinetic_mesh

from .common import constants as CONST

def interp_fvrvxx(fa, mesh_a : kinetic_mesh, mesh_b : kinetic_mesh, do_warn=None, debug=0, correct=1, debug_flag = 0): #NOTE Debug flag added to mark specific function calls, remove later
    
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

    prompt = 'INTERP_FVRVXX => '

    v_scale = np.sqrt(mesh_b.Tnorm / mesh_a.Tnorm) # velocity ratio (scales velocities from mesh_a to mesh_b)
    
    # Check shape agreement for fa
    if fa.shape != (mesh_a.vr.size, mesh_a.vx.size, mesh_a.x.size):
        raise Exception('fa (' + str(fa.shape) + ') does not have shape (vra, vxa, xa)' + str((mesh_a.vr.size, mesh_a.vx.size, mesh_a.x.size)))


    # --- Get interpolation Bounds ---

    ii = np.where((min(mesh_a.vr) <= v_scale*mesh_b.vr) & (v_scale*mesh_b.vr <= max(mesh_a.vr)))[0]
    if len(ii) < 1:
        raise Exception('No values of Vrb are within range of Vra')
    vr_start, vr_end = ii[0], ii[-1]

    ii = np.where((min(mesh_a.vx) <= v_scale*mesh_b.vx) & (v_scale*mesh_b.vx <= max(mesh_a.vx)))[0]
    if ii.size < 1:
        raise Exception('No values of Vxb are within range of Vxa')
    vx_start, vx_end = ii[0], ii[-1]

    ii = np.where((min(mesh_a.x) <= mesh_b.x) & (mesh_b.x <= max(mesh_a.x)))[0]
    if ii.size < 1:
        raise Exception('No values of Xb are within range of Xa')
    x_start, x_end = ii[0], ii[-1]

    fb = np.zeros((mesh_b.vr.size, mesh_b.vx.size, mesh_b.x.size))


    # --- Generate differentials ---

    vdiff_a = VSpace_Differentials(mesh_a.vr, mesh_a.vx)
    vdiff_b = VSpace_Differentials(mesh_b.vr, mesh_b.vx)
    

    # --- Compute Weights ---

    if debug:
        print(prompt+'computing new weight')

    # NOTE Removed saving weights temporarily, re-implement later

    # NOTE This is slightly more confusing than the original method, but should be more efficient
    # Set area contributions to Weight array
    # Get arrays of element-wise min/max values for vr and vx, comparing mesh_a and mesh_b
    vr_min = np.maximum(v_scale*vdiff_b.vr_left_bound[:, np.newaxis, np.newaxis, np.newaxis],
                                vdiff_a.vr_left_bound[np.newaxis, np.newaxis, :, np.newaxis])
    vr_max = np.minimum(v_scale*vdiff_b.vr_right_bound[:, np.newaxis, np.newaxis, np.newaxis],
                                vdiff_a.vr_right_bound[np.newaxis, np.newaxis, :, np.newaxis])
    
    vx_min = np.maximum(v_scale*vdiff_b.vx_left_bound[np.newaxis, :, np.newaxis, np.newaxis],
                                vdiff_a.vx_left_bound[np.newaxis, np.newaxis, np.newaxis, :])
    vx_max = np.minimum(v_scale*vdiff_b.vx_right_bound[np.newaxis, :, np.newaxis, np.newaxis],
                                vdiff_a.vx_right_bound[np.newaxis, np.newaxis, np.newaxis, :])

    # Calculate weights
    condition = (vr_max > vr_min) & (vx_max > vx_min)
    weight_value = 2*np.pi*(vr_max**2 - vr_min**2)*(vx_max - vx_min) / (vdiff_b.dvr_vol[:, np.newaxis, np.newaxis, np.newaxis]*vdiff_b.dvx[np.newaxis, :, np.newaxis, np.newaxis])
    weight = np.where(condition, weight_value, 0)

    # Convert to 2D
    weight = np.reshape(weight, (mesh_b.vr.size*mesh_b.vx.size, mesh_a.vr.size*mesh_a.vx.size), order = 'F')


    # --- Correct fb so that it has the same Wx and E moments as fa ---

    if correct:

        # --- Compute Desired Moments ---

        # Determine fb distribution on mesh_a.x grid from weight array
        fa_reshaped = np.reshape(fa, (mesh_a.vr.size*mesh_a.vx.size, mesh_a.x.size), order = 'F')
        fb_on_xa = np.matmul(weight, fa_reshaped)

        #   Compute desired vx_moment and energy_moments of fb, but on the xa grid
        vx_moment_on_xa = np.zeros(mesh_a.x.size)
        energy_moment_on_xa = np.zeros(mesh_a.x.size)

        for k in range(mesh_a.x.size):
            density_a = np.sum(vdiff_a.dvr_vol*(np.matmul(fa[:,:,k], vdiff_a.dvx)))
            if density_a > 0:
                vx_moment_on_xa[k] = np.sqrt(mesh_a.Tnorm)*np.sum(vdiff_a.dvr_vol*(np.matmul(fa[:,:,k], (mesh_a.vx*vdiff_a.dvx)))) / density_a
                energy_moment_on_xa[k] = mesh_a.Tnorm*np.sum(vdiff_a.dvr_vol*(np.matmul((vdiff_a.vmag_squared*fa[:,:,k]), vdiff_a.dvx))) / density_a

        # Compute desired moments on xb grid
        target_vx = np.zeros(mesh_b.x.size)
        target_energy = np.zeros(mesh_b.x.size)

        for k in range(x_start, x_end+1):
            position = np.maximum(locate(mesh_a.x, mesh_b.x[k]), 0)
            kr = np.minimum(position+1, mesh_a.x.size-1)
            kl = np.minimum(position, kr-1)

            interp_fraction = (mesh_b.x[k] - mesh_a.x[kl]) / (mesh_a.x[kr] - mesh_a.x[kl])
            fb[:,:,k] = np.reshape((fb_on_xa[:,kl] + interp_fraction*(fb_on_xa[:,kr] - fb_on_xa[:,kl])), fb[:,:,k].shape, order='F')
            target_vx[k] = vx_moment_on_xa[kl] + interp_fraction*(vx_moment_on_xa[kr] - vx_moment_on_xa[kl])
            target_energy[k] = energy_moment_on_xa[kl] + interp_fraction*(energy_moment_on_xa[kr] - energy_moment_on_xa[kl])

        #   Process each spatial location
        for k in range(mesh_b.x.size):
            if target_energy[k] is None:
                continue

            allow_neg = 0

            #   Compute nb, Wxb, and Eb - these are the current moments of fb

            nb = np.sum(vdiff_b.dvr_vol*(fb[:,:,k] @ vdiff_b.dvx))
            if nb <= 0:
                continue

            goto_correct = True
            while goto_correct:
                
                # --- Adjust fb for desired weights ---

                fb[:,:,k], s = compensate_distribution(fb[:,:,k], vdiff_b, mesh_b.vr, mesh_b.vx, np.sqrt(mesh_b.Tnorm), target_vx[k], target_energy[k], nb=nb, assume_pos=False)
                
                # print("goto", goto_correct)
                goto_correct = (s < 1)

    if(debug_flag != 0):
        ii = np.nonzero(fb.reshape(fb.size, order='F'))
        print("fSHAnz", ii)
        input()

    if do_warn != None:

        #   Test Boundaries:

        #   vr_start & vr_end 

        big = np.max(fb)

        vr_start_error = 0
        vr_end_error = 0
        if (vr_start > 0) or (vr_end < mesh_b.vr.size-1):
            for k in range(x_start, x_end+1):
                for j in range(vx_start, vx_end+1):
                    if (vr_start_error == 0) and (vr_start > 0) and (fb[vr_start,j,k] > do_warn*big):
                        warn('Non-zero value of fb detected at min(Vra) boundary')
                        vr_start_error = 1
                    if (vr_end_error == 0) and (vr_end < mesh_b.vr.size-1) and (fb[vr_end,j,k] > do_warn*big): # fixed capitalization
                        warn('Non-zero value of fb detected at max(Vra) boundary')
                        vr_end_error = 1

        #   vx_start & vx_end

        vx_start_error = 0
        vx_end_error = 0
        if (vx_start > 0) or (vx_end < mesh_b.vx.size-1):
            for k in range(x_start, x_end+1):
                for i in range(vr_start, vr_end+1):
                    if (vx_start_error == 0) and (vx_start > 0) and (fb[i,vx_start,k] > do_warn*big):
                        warn('Non-zero value of fb detected at min(Vxa) boundary')
                        vx_start_error=1
                    if (vx_end_error == 0) and (vx_end < mesh_b.vx.size-1) and (fb[i,vx_end,k] > do_warn*big): # fixed capitalization
                        warn('Non-zero value of fb detected at max(Vxa) boundary')
                        vx_end_error=1

        #   x_start & x_end

        x_start_error = 0
        x_end_error = 0
        if (x_start > 0) or (x_end < mesh_b.x.size-1):
            for i in range(vr_start, vr_end+1):
                for j in range(vx_start, vx_end+1):
                    if (x_start_error == 0) and (x_start > 0) and (fb[i,j,x_start] > do_warn*big):
                        warn('Non-zero value of fb detected at min(Xa) boundary')
                        x_start_error = 1
                    if (x_end_error == 0) and (x_end < mesh_b.x.size-1) and (fb[i,j,x_end] > do_warn*big):
                        warn('Non-zero value of fb detected at max(Xa) boundary')
                        x_end_error = 1

    #   Rescale

    tot_a = np.zeros(mesh_a.x.size)
    for k in range(mesh_a.x.size):
        tot_a[k] = np.sum(vdiff_a.dvr_vol*(fa[:,:,k] @ vdiff_a.dvx))
    tot_b = np.zeros(mesh_b.x.size)
    tot_b[x_start:x_end+1] = interpolate.interp1d(mesh_a.x,tot_a,fill_value="extrapolate")(mesh_b.x[x_start:x_end+1])
    ii = np.where(fb>0)
    if ii[0].size > 0: # replaced fb with ii
        min_tot = np.min(np.array(fb[ii])) #(np.array([fb[tuple(i)] for i in ii]))
        for k in range(x_start, x_end+1):
            tot = np.sum(vdiff_b.dvr_vol*(fb[:,:,k] @ vdiff_b.dvx))
            if tot > min_tot:
                if debug:
                    print(prompt+'Density renormalization factor ='+sval(tot_b[k]/tot))
                fb[:,:,k] = fb[:,:,k]*tot_b[k]/tot

    if debug:

        #   Compute Vtha, Vtha2, Vthb and Vthb2
        mass = 1*CONST.H_MASS # mu*hydrogen mass
        vth1 = np.sqrt((2*CONST.Q*mesh_a.Tnorm) / mass)
        vth2 = np.sqrt((2*CONST.Q*mesh_b.Tnorm) / mass)

        #   na, Uxa, Ta
        na = np.zeros(mesh_a.x.size)
        Uxa = np.zeros(mesh_a.x.size)
        Ta = np.zeros(mesh_a.x.size)
        vr2vx2_ran2 = np.zeros((mesh_a.vr.size,mesh_a.vx.size)) # fixed np.zeros() call

        for k in range(mesh_a.x.size):
            na[k] = np.sum(vdiff_a.dvr_vol*(fa[:,:,k] @ vdiff_a.dvx)) # fixed capitalization
            if na[k] > 0:
                Uxa[k] = vth1*np.sum(vdiff_a.dvr_vol*(fa[k,:,:] @ (mesh_a.vx*vdiff_a.dvx)))/na[k]
                for i in range(mesh_a.vr.size):
                    vr2vx2_ran2[i,:] = mesh_a.vr[i]**2 + (mesh_a.vx - Uxa[k]/vth1)**2 # fixed capitalization
                Ta[k] = mass*(vth1**2)*np.sum(vdiff_a.dvr_vol*((vr2vx2_ran2*fa[:,:,k]) @ vdiff_a.dvx))/(3*CONST.Q*na[k])

        #   nb, Uxb, Tb
        nb = np.zeros(mesh_b.x.size)
        Uxb = np.zeros(mesh_b.x.size)
        Tb = np.zeros(mesh_b.x.size)
        vr2vx2_ran2 = np.zeros((mesh_b.vr.size,mesh_b.vx.size)) # fixed np.zeros() call

        for k in range(mesh_b.x.size):
            nb[k] = np.sum(vdiff_b.dvr_vol*(fb[:,:,k] @ vdiff_b.dvx))
            if nb[k] > 0:
                Uxb[k] = vth2*np.sum(vdiff_b.dvr_vol*(fb[:,:,k] @ (mesh_b.vx*vdiff_b.dvx)))/nb[k] # fixed typo
                for i in range(mesh_b.vr.size):
                    vr2vx2_ran2[i,:] = mesh_b.vr[i]**2 + (mesh_b.vx - Uxb[k]/vth2)**2 # fixed capitalization
                Tb[k] = mass*(vth2**2)*np.sum(vdiff_b.dvr_vol*((vr2vx2_ran2*fb[:,:,k]) @ vdiff_b.dvx))/(3*CONST.Q*nb[k])

        #   Plotting stuff was here in the original code
        #   May be added later, but has been left out for now

    return fb
