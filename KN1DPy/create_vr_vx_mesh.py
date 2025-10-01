import numpy as np
from numpy.typing import NDArray

from .utils import reverse

# sets up optimum Vr and Vx velocity space mesh for Kinetic_Neutrals procedure 
# Input: 
#   nv - Integer, number of elements desired in vr mesh
#   Ti - arrray, Ti profile
#   E0 - array, energy where a velocity is desired ( optional )
#   Tmax - float, ignore Ti above this value
#
# Gwendolyn Galleher 

def create_vr_vx_mesh(nv: int, Ti: NDArray, E0: NDArray = np.array([0.0]), Tmax: float = 0.0) -> tuple[NDArray, NDArray, float] :
    Ti = np.array(Ti) 
    Ti = np.concatenate([Ti, E0[E0>0]])
    if Tmax > 0:
        ii = np.where(Ti < Tmax)
        Ti = Ti[ii]
    
    maxTi = Ti.max()
    minTi = Ti.min()
    Tnorm = np.nanmean(Ti)
    vmax = 3.5
    if (maxTi-minTi) <= (0.1*maxTi):
        v = np.arange(nv+1)*vmax/nv
    else:
        g = 2*nv*np.sqrt(minTi/maxTi) / (1 - np.sqrt(minTi/maxTi))
        b = vmax / (nv*(nv + g))
        v = (g*b)*np.arange(nv+1) + b*(np.arange(nv+1)**2)

    # Option: add velocity bins corresponding to E0     
    v0 = 0
    for k in range(np.size(E0)):
        if E0[k] > 0.0:
            v0 = np.sqrt(E0[k]/Tnorm)
            ii = np.argwhere(v > v0).T[0]
            if np.size(ii) > 0:
                v = np.concatenate([v[0:ii[0]], [v0], v[ii[0]:]])
            else: 
                v = np.concatenate([v, v0])
        
    vr = v[1:]
    vx = np.concatenate([-reverse(vr), vr]) 

    return vx,vr,Tnorm
