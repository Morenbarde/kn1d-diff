import numpy as np
import torch

from ..utils import poly
from ..torch_utils import poly_torch

def sigmav_p_p_hp(Te: torch.Tensor):
    '''
    Returns maxwellian averaged <sigma V) for electron impact 
    dissociation of molecular hydrogen ions resulting in two protons. 
    Coefficients are taken from 
    
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
            -3.746192301092e+1, 
            1.559355031108e+1, 
            -6.693238367093e+0, 
            1.981700292134e+0, 
            -4.044820889297e-1, 
            5.352391623039e-2, 
            -4.317451841436e-3, 
            1.918499873454e-4, 
            -3.591779705419e-6
        ], dtype=Te.dtype, device=Te.device)
    
    # Ensure 0.1 < Te < 2.01e4
    Te = torch.clamp(Te, 0.1, 2.01e4)
    
    return torch.exp(poly_torch(torch.log(Te), b))*1e-6