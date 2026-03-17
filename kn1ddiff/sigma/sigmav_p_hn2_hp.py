import numpy as np
import torch

from ..utils import poly
from ..torch_utils import poly_torch

def sigmav_p_hn2_hp(Te: torch.Tensor):
    '''
    Returns maxwellian averaged <sigma V) for electron impact dissociation 
    of molecular hydrogen ions resulting in one proton and one H atom 
    in the n=2 state. Coefficients are taken from 
    
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
            -3.408905929046e+1, 
            1.573560727511e+1, 
            -6.992177456733e+0, 
            1.852216261706e+0, 
            -3.130312806531e-1, 
            3.383704123189e-2, 
            -2.265770525273e-3, 
            8.565603779673e-5, 
            -1.398131377085e-6
        ], dtype=Te.dtype, device=Te.device)
    
    # Ensure 0.1 < Te < 2.01e4
    Te = torch.clamp(Te, 0.1, 2.01e4)
    
    return torch.exp(poly_torch(torch.log(Te), b))*1e-6