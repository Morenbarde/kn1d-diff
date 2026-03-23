import numpy as np
import torch

from ..utils import poly
from ..torch_utils import poly_torch, dclamp

def sigmav_h1s_h2s_hh(Te: torch.Tensor):
    '''
    Computes maxwellian averaged <sigma V) for electron impact
    dissociation of molecular hydrogen resulting in one H atom in
    the 1s state and one H atom in the 2s state. Coefficients are taken from 

        Janev, "Elementary Processes in Hydrogen-Helium Plasmas",
        Springer-Verlag, 1987, p.259.

    Parameters
    ----------
    Te : ndarray or float
        electron temperature (eV)

    Returns
    -------
        ndarray
            Sigma V for 0.1 < Te < 2e4. (m^3/s)
    '''

    # Te = np.asarray(Te)

    b = torch.tensor([-3.454175591367e+1,
                    1.412655911280e+1,
                    -6.004466156761e+0,
                    1.589476697488e+0,
                    -2.775796909649e-1,
                    3.152736888124e-2,
                    -2.229578042005e-3,
                    8.890114963166e-5,
                    -1.523912962346e-6],
                dtype=Te.dtype, device=Te.device)
    
    # Ensure 0.1 < Te < 2.01e4
    # Te = torch.clamp(Te, 0.1, 2.01e4)
    Te = dclamp(Te, 0.1, 2.01e4)

    result = torch.exp(poly_torch(torch.log(Te), b))*1e-6
    return result 
