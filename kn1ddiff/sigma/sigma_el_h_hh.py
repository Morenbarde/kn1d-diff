import numpy as np
import torch

from ..utils import poly
from ..torch_utils import poly_torch, dclamp

def sigma_el_h_hh(E: torch.Tensor): 
    '''
    Computes momentum transfer cross section for elastic collisions of H onto H2 
    for specified energy of H. Data are taken from 

        Janev, "Atomic and Molecular Processes in Fusion Edge Plasmas", Chapter 11 - 
        Elastic and Related Cross Sections for Low-Energy Collisions among Hydrogen and 
        Helium Ions, Neutrals, and Isotopes  by D.R. Sdhultz, S. Yu. Ovchinnikov, and S.V.
        Passovets, page 305.

    Parameters
    ----------
    E : ndarray or float
        energy of H atom (target H2 molecule is at rest)

    Returns
    -------
        ndarray
            Sigma for 0.03 < E < 1e4. For E outside this range, 
            the value of Sigma at the 0.03 or 1e4 eV boundary is returned. (m^-2)
    '''

    # Ensure 0.03e0 < E < 1.01e4
    # E = torch.clamp(E, 0.03e0, 1.01e4)
    E = dclamp(E, 0.03e0, 1.01e4)
    a = torch.tensor([-3.495671e1, -4.062257e-1, -3.820531e-2, -9.404486e-3, 3.963723e-4], dtype=E.dtype, device=E.device)
    
    result = torch.exp(poly_torch(torch.log(E), a)) * 1e-4
    return result