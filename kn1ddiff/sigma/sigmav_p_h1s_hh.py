import numpy as np
import torch

from ..utils import poly
from ..torch_utils import poly_torch, dclamp

def sigmav_p_h1s_hh(Te: torch.Tensor):
    '''
    Returns maxwellian averaged <sigma V) for electron impact ionization 
    and disociation of Molecular hydrogen resulting in one proton and 
    one H atom in the 1s state. Coefficients are taken from 
    
        Janev, "Elementary Processes in Hydrogen-Helium Plasmas",
        Springer-Verlag, 1987, p.260.

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
    
    b = torch.tensor([
            -3.834597006782e+1, 
            1.426322356722e+1, 
            -5.826468569506e+0, 
            1.727940947913e+0, 
            -3.598120866343e-1, 
            4.822199350494e-2, 
            -3.909402993006e-3, 
            1.738776657690e-4, 
            -3.252844486351e-6
        ], dtype=Te.dtype, device=Te.device)

    
    # Ensure 0.1 < Te < 2.01e4
    # Te = torch.clamp(Te, 0.1, 2.01e4)
    Te = dclamp(Te, 0.1, 2.01e4)

    result = torch.exp(poly_torch(torch.log(Te), b))*1e-6
    return result