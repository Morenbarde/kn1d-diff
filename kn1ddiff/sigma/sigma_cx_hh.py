import numpy as np
import torch

from ..utils import poly
from ..torch_utils import dclamp, poly_torch

def sigma_cx_hh(E: torch.Tensor):
    '''
    Computes charge exchange cross section for molecular hydrogen. Data are taken the polynomial fit in
        Janev, "Elementary Processes in Hydrogen-Helium Plasmas", Springer-Verlag, 1987, p.253.

    Parameters
    ----------
        E : ndarray or float
            energy of molecule corresponding to the relative velocity between molecule and molecular ion. (eV)

    Returns
    -------
        ndarray
            sigma_CX for 0.1 < E < 2e4 (m^-2)
    '''

    # E = np.asarray(E)

    # Ensure 0.1 < E < 2.01e4
    E = dclamp(E, 0.1, 2.01e4)
    alpha = torch.tensor([  
                    -3.427958758517e+01, -7.121484125189e-02, 4.690466187943e-02,
                    -8.033946660540e-03, -2.265090924593e-03,-2.102414848737e-04,
                    1.948869487515e-04, -2.208124950005e-05, 7.262446915488e-07],
                    dtype=E.dtype, device=E.device)

    return torch.exp(poly_torch(torch.log(E), alpha))*1e-4