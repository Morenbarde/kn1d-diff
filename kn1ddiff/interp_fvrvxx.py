import numpy as np
from warnings import warn

import torch
from .torch_utils import *

from .make_dvr_dvx import VSpace_Differentials
from .create_shifted_maxwellian import compensate_distribution
from .utils import sval, locate, Bound, interp_1d, get_config
from .kinetic_mesh import KineticMesh

def _get_interpolation_bounds(a, b, a_name="a", b_name="b"):
    '''
    Internal method for usage in interp_fvrvxx
    Generates bounds for where b is within the range of a

    Parameters
    ----------
        a : ndarray
            array that determines bounds of b
        b : ndarray
            array being bounded
        a_name, b_name : str (optional)
            variable names for exception message

    Returns
    -------
        tuple : (int, int)
            start, end of the interpolation range
    
    Raises
    -------
        Exception if interpolation range is 0
    '''

    ii = np.where((min(a) <= b) & (b <= max(a)))[0]
    if len(ii) < 1:
        raise Exception(f"No values of {b_name} are within range of {a_name}")
    return Bound(ii[0], ii[-1])


def _test_bounds(fb, test_bound : Bound, var_len, test_axis, iter_bound1 : Bound, iter_bound2 : Bound, do_warn, var_name="a"):
    '''
    Internal method for usage in interp_fvrvxx
    Tests boundaries for vr, vx, x

    Parameters
    ----------
        fb : ndarray
            3D array, distribution function
        test_bound : Bound
            boundary being tested
        var_len : int
            Length of the variable distribution whose bound is being tested
        test_axis : int
            Determines axis for slicing fb, 0, 1, or 2
        iter_bound1, iter_bound2 : Bound
            Boundaries being tested over
        do_warn : float
            Acceptable truncation level
        var_name : str (optional)
            variable name for warning message
    
    Issues
    -------
        Warning if 0 found at boundary edge
    '''

    big = torch.max(fb)
    start_error = 0
    end_error = 0
    if (test_bound.start > 0) or (test_bound.end < var_len-1):
        iter_slice1 = iter_bound1.slice(0,1)
        iter_slice2 = iter_bound2.slice(0,1)
        if test_axis == 0:
            min_slice = fb[test_bound.start, iter_slice1, iter_slice2]
            max_slice = fb[test_bound.end, iter_slice1, iter_slice2]
        elif test_axis == 1:
            min_slice = fb[iter_slice1, test_bound.start, iter_slice2]
            max_slice = fb[iter_slice1, test_bound.end, iter_slice2]
        elif test_axis == 2:
            min_slice = fb[iter_slice1, iter_slice2, test_bound.start]
            max_slice = fb[iter_slice1, iter_slice2, test_bound.end]
        else:
            raise Exception("Invalid test axis")
        
        if (start_error == 0) and (test_bound.start > 0) and torch.any(min_slice > do_warn*big):
            warn(f"Non-zero value of fb detected at min({var_name}) boundary")
        if (end_error == 0) and (test_bound.end < var_len-1) and torch.any(max_slice > do_warn*big):
            warn(f"Non-zero value of fb detected at max({var_name}) boundary")


def interp_fvrvxx(fa: np.ndarray, mesh_a : KineticMesh, mesh_b : KineticMesh, do_warn=None, debug=False, correct=1):
    '''
    Interpolates distribution functions used by kinetic neutral procedures

    Parameters
    ----------
        fa : ndarray
            Input distribution function, 3D array of shape (vra, vxa, xa)
        mesh_a : KineticMesh
            Mesh information for input distribution
        mesh_b : KineticMesh
            Mesh information for desired output distribution
        do_warn : float or None (optional)
            Accebtable truncation level. If None, warnings are not generated
                For interpolations outside the phase space set by
                (Vra, Vxa, Xa), the values of fb are set to zero.
                This may not be acceptable. A test is performed on
                fb at the boundaries. If fb at the boundaries is greater
                than do_warn times the maximum value of fb,
                a warning message is generated.
        debug : bool
            If True, generate debug statements
            
    Returns
    -------
        fb : ndarray
            Interpolated distribution function, scaled if necessary to make its 
            digital integral over all velocity space equal to that of fa
            3D array of of shape (vrb, vxb, xb)
    '''

    prompt = 'INTERP_FVRVXX => '

    nvr_b, nvx_b, nx_b = mesh_b.vr.numel(), mesh_b.vx.numel(), mesh_b.x.numel()
    nvr_a, nvx_a, nx_a = mesh_a.vr.numel(), mesh_a.vx.numel(), mesh_a.x.numel()

    v_scale = torch.sqrt(mesh_b.Tnorm / mesh_a.Tnorm) # velocity ratio (scales velocities from mesh_a to mesh_b)
    
    # Check shape agreement for fa
    if fa.shape != (nvr_a, nvx_a, nx_a):
        raise Exception('fa (' + str(fa.shape) + ') does not have shape (vra, vxa, xa)' + str((nvr_a, nvx_a, nx_a)))


    # --- Get interpolation Bounds ---

    get_range = lambda a, b : torch.where((min(a) <= b) & (b <= max(a)))[0]

    vr_bound = _get_interpolation_bounds(mesh_a.vr, v_scale*mesh_b.vr, "Vra", "Vrb")
    vx_bound = _get_interpolation_bounds(mesh_a.vx, v_scale*mesh_b.vx, "Vxa", "Vxb")
    x_bound = _get_interpolation_bounds(mesh_a.x, mesh_b.x, "Xa", "Xb")

    fb = torch.zeros((nvr_b, nvx_b, nx_b), dtype=fa.dtype, device=fa.device)


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
    vr_min = torch.maximum(v_scale*vdiff_b.vr_left_bound[:, None, None, None],
                                vdiff_a.vr_left_bound[None, None, :, None])
    vr_max = torch.minimum(v_scale*vdiff_b.vr_right_bound[:, None, None, None],
                                vdiff_a.vr_right_bound[None, None, :, None])
    
    vx_min = torch.maximum(v_scale*vdiff_b.vx_left_bound[None, :, None, None],
                                vdiff_a.vx_left_bound[None, None, None, :])
    vx_max = torch.minimum(v_scale*vdiff_b.vx_right_bound[None, :, None, None],
                                vdiff_a.vx_right_bound[None, None, None, :])

    # Calculate weights
    condition = (vr_max > vr_min) & (vx_max > vx_min)
    weight_value = 2*torch.pi*(vr_max**2 - vr_min**2)*(vx_max - vx_min) / (vdiff_b.dvr_vol[:, None, None, None]*vdiff_b.dvx[None, :, None, None])
    weight = torch.where(condition, weight_value, 0)

    # Convert to 2D
    # weight = np.reshape(weight, (nvr_b*nvx_b, nvr_a*nvx_a), order = 'F')
    weight = torch_reshape_fortran(weight, (nvr_b*nvx_b, nvr_a*nvx_a))


    # --- Correct fb so that it has the same Wx and E moments as fa ---

    if correct:

        # --- Compute Desired Moments ---

        # Determine fb distribution on mesh_a.x grid from weight array
        # fa_reshaped = np.reshape(fa, (nvr_a*nvx_a, nx_a), order = 'F')
        fa_reshaped = torch_reshape_fortran(fa, (nvr_a*nvx_a, nx_a))
        fb_on_xa = weight @ fa_reshaped

        #   Compute desired vx_moment and energy_moments of fb, but on the xa grid
        vx_moment_on_xa = torch.zeros_like(mesh_a.x)
        energy_moment_on_xa = torch.zeros_like(mesh_a.x)

        epsilon = 1e-8

        for k in range(nx_a):
            density_a = torch.sum(vdiff_a.dvr_vol*(fa[:,:,k] @ vdiff_a.dvx))
            # if density_a > 0:
            vx_moment_on_xa[k] = torch.sqrt(mesh_a.Tnorm)*torch.sum(vdiff_a.dvr_vol*(fa[:,:,k] @ (mesh_a.vx*vdiff_a.dvx))) / (density_a+epsilon)
            energy_moment_on_xa[k] = mesh_a.Tnorm*torch.sum(vdiff_a.dvr_vol*(torch.matmul((vdiff_a.vmag_squared*fa[:,:,k]), vdiff_a.dvx))) / (density_a+epsilon)

        # Compute desired moments on xb grid
        target_vx = torch.zeros_like(mesh_b.x)
        target_energy = torch.zeros_like(mesh_b.x)

        for k in range(x_bound.start, x_bound.end+1):
            position = torch.maximum(torch_locate(mesh_a.x, mesh_b.x[k]), torch.tensor(0, dtype=torch.long, device=fa.device))
            kr = torch.minimum(position+1, torch.tensor(nx_a-1, dtype=torch.long, device=fa.device))
            kl = torch.minimum(position, kr-1)

            interp_fraction = (mesh_b.x[k] - mesh_a.x[kl]) / (mesh_a.x[kr] - mesh_a.x[kl])
            # fb[:,:,k] = np.reshape((fb_on_xa[:,kl] + interp_fraction*(fb_on_xa[:,kr] - fb_on_xa[:,kl])), fb[:,:,k].shape, order='F')
            fb[:,:,k] = torch_reshape_fortran((fb_on_xa[:,kl] + interp_fraction*(fb_on_xa[:,kr] - fb_on_xa[:,kl])), fb[:,:,k].shape)
            target_vx[k] = vx_moment_on_xa[kl] + interp_fraction*(vx_moment_on_xa[kr] - vx_moment_on_xa[kl])
            target_energy[k] = energy_moment_on_xa[kl] + interp_fraction*(energy_moment_on_xa[kr] - energy_moment_on_xa[kl])


        #   Process each spatial location
        for k in range(nx_b):
            # if target_energy[k] is None:
            #     continue

            #   Compute nb, Wxb, and Eb - these are the current moments of fb

            nb = torch.sum(vdiff_b.dvr_vol*(torch.matmul(fb[:,:,k], vdiff_b.dvx)))
            # if nb <= 0:
            #     continue


            while True:
                
                # --- Adjust fb for desired weights ---

                fb[:,:,k], s = compensate_distribution(fb[:,:,k], vdiff_b, mesh_b.vr, mesh_b.vx, np.sqrt(mesh_b.Tnorm), target_vx[k], target_energy[k], nb=(nb+epsilon), assume_pos=False)
                if s >= 1:
                    break


    # --- Test Boundaries ---

    if do_warn != None:
        # vr_bound
        _test_bounds(fb, vr_bound, nvr_b, 0, vx_bound, x_bound, do_warn, var_name="Vra")
        # vx_bound
        _test_bounds(fb, vx_bound, nvx_b, 1, vr_bound, x_bound, do_warn, var_name="Vxa")
        # x_bound
        _test_bounds(fb, x_bound, nx_b, 2, vr_bound, vx_bound, do_warn, var_name="Xa")


    # --- Rescale ---

    tot_a = torch.zeros_like(mesh_a.x)
    for k in range(nx_a):
        tot_a[k] = torch.sum(vdiff_a.dvr_vol*(torch.matmul(fa[:,:,k], vdiff_a.dvx)))
        
    tot_b = torch.zeros_like(mesh_b.x)
    # tot_b[x_bound.slice(0,1)] = interp_1d(mesh_a.x, tot_a, mesh_b.x[x_bound.slice(0,1)], fill_value="extrapolate")
    tot_b[x_bound.slice(0,1)] = torch_interp1d(mesh_b.x[x_bound.slice(0,1)], mesh_a.x, tot_a)

    ii = torch.where(fb > 0)
    if ii[0].numel() > 0:
        min_tot = torch.min(fb[ii].detach().clone())
        for k in x_bound.range():
            tot = torch.sum(vdiff_b.dvr_vol*(torch.matmul(fb[:,:,k], vdiff_b.dvx)))
            if tot > min_tot:
                if debug:
                    print(prompt + 'Density renormalization factor =' + sval(tot_b[k] / tot))
                fb[:,:,k] = fb[:,:,k]*tot_b[k]/tot


    return fb

