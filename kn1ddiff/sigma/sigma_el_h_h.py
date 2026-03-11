import numpy as np 
import torch

from ..utils import poly
from ..torch_utils import poly_torch, dclamp

def sigma_el_h_h(E: torch.Tensor, vis = False):
    '''
    Computes momentum transfer cross section for elastic collisions of H onto H
    for specified energy of H. Data are taken from 

        Janev, "Atomic and Molecular Processes in Fusion Edge Plasmas", Chapter 11 - 
        Elastic and Related Cross Sections for Low-Energy Collisions among Hydrogen and 
        Helium Ions, Neutrals, and Isotopes  by D.R. Sdhultz, S. Yu. Ovchinnikov, and S.V.
        Passovets, page 305.

    Parameters
    ----------
    E : ndarray or float
        energy of H atom (target H atom is at rest)
    vis : bool, defaul=False
        if true, then return viscosity cross section instead of momentum transfer cross section 
    
    Returns
    -------
        ndarray
            Sigma for 0.03 < E < 1e4. For E outside this range, 
            the value of Sigma at the 0.03 or 1e4 eV boundary is returned. (m^-2)
    '''

    # Ensure 0.03e0 < E < 1.01e4
    # E = torch.clamp(E, 0.03e0, 1.01e4)
    E = dclamp(E, 0.03e0, 1.01e4)

    if vis: 
        # calculates viscosity cross section
        a = torch.tensor([ -3.344860e1, -4.238982e-1, -7.477873e-2, -7.915053e-3, -2.686129e-4], dtype=E.dtype, device=E.device)
    else: 
        # calculates momentum transfer cross section
        a = torch.tensor([ -3.330843e1, -5.738374e-1, -1.028610e-1, -3.920980e-3, 5.964135e-4], dtype=E.dtype, device=E.device)
    
    result = torch.exp(poly_torch(torch.log(E), a)) * 1e-4
    return result


