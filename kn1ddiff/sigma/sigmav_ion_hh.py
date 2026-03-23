import numpy as np
import torch

from ..utils import poly
from ..torch_utils import poly_torch, dclamp

def sigmav_ion_hh(Te: torch.Tensor):
    '''
    Returns maxwellian averaged <sigma V) for electron impact
    ionization of molecular hydrogen. Coefficients are taken from 
    
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

    b = torch.tensor([-3.568640293666e+1,
                        1.733468989961e+1, 
                        -7.767469363538e+0, 
                        2.211579405415e+0, 
                        -4.169840174384e-1, 
                        5.088289820867e-2, 
                        -3.832737518325e-3, 
                        1.612863120371e-4, 
                        -2.893391904431e-6],
                    dtype=Te.dtype, device=Te.device)

    
    # Ensure 0.1 < Te < 2.01e4
    # Te = torch.clamp(Te, 0.1, 2.01e4)
    Te = dclamp(Te, 0.1, 2.01e4)

    result = torch.exp(poly_torch(torch.log(Te), b))*1e-6
    return result
