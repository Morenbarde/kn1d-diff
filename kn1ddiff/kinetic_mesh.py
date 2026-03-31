import numpy as np 
from numpy.typing import NDArray
import torch

from .utils import get_config, interp_1d, reverse
from .torch_utils import *
from .sigma.sigmav_ion_h0 import sigmav_ion_h0
from .sigma.sigmav_cx_h0 import sigmav_cx_h0
from .sigma.sigmav_ion_hh import sigmav_ion_hh
from .sigma.sigmav_h1s_h1s_hh import sigmav_h1s_h1s_hh
from .sigma.sigmav_h1s_h2s_hh import sigmav_h1s_h2s_hh
from .sigma.sigmav_cx_hh import sigmav_cx_hh
from .sigma.collrad_sigmav_ion_h0 import collrad_sigmav_ion_h0
from .johnson_hinnov import Johnson_Hinnov

from .common import constants as CONST


class KineticMesh:
    '''
    Mesh data for kinetic_neutrals procedure

    Attributes
    ----------
        mesh_type : str
            Type of mesh data
            - 'h' for atomic
            - 'h2' for molecular
        x : ndarray
            spatial coordinates
        vx : ndarray
            axial velocity coordinates
        vr : ndarray
            radial velocity coordinates
        Ti : ndarray
            ion temperature profile interpolated over x
        Te : ndarray
            electron temperature profile interpolated over x
        ne : ndarray
            density profile interpolated over x  
        PipeDia : ndarray
            effective pipe diameter interpolated over x
        Tnorm : float
            Average ion temperature
    '''

    def __init__( #NOTE Simplify this later, consider using class inheritance
            self, 
            mesh_type   : str, #'h' for kinetic_h_mesh, 'h2' for kinetic_h2_mesh
            mu          : int, 
            x           : NDArray,
            Ti          : NDArray,
            Te          : NDArray, 
            n           : NDArray, 
            PipeDia     : NDArray,
            jh          : Johnson_Hinnov = None,
            E0          : NDArray = None, 
            fctr        : float   = 1.0,
            xH          : NDArray = None):


        if xH is None:
            print("generating full kinetic_" + mesh_type + "_mesh")
            self.init_full(mesh_type, mu, x, Ti, Te, n, PipeDia, jh, E0, fctr)
        else:
            print("generating lite kinetic_" + mesh_type + "_mesh")
            self.init_lite(mesh_type, mu, x, xH, Ti, Te, n, PipeDia, jh, E0, fctr)

        


    def init_full(
            self, 
            mesh_type   : str, #'h' for kinetic_h_mesh, 'h2' for kinetic_h2_mesh
            mu          : int, 
            x           : NDArray,
            Ti          : NDArray,
            Te          : NDArray, 
            n           : NDArray, 
            PipeDia     : NDArray,
            jh          : Johnson_Hinnov = None,
            E0          : NDArray = None, 
            fctr        : float   = 1.0,):
        
        if E0 is None:
            E0 = torch.tensor([0.0], device=x.device, dtype=x.dtype)

        #Get mesh size from config file
        nv = get_config()["kinetic_" + mesh_type]["mesh_size"]

        # estimate Interaction rate with side walls
        #NOTE Commented gamma_wall calculations here, revisit later

        if mesh_type == 'h':
            # Estimate total reaction rate for destruction of hydrogen atoms and for interation with side walls
            react_rate = n*sigmav_ion_h0(Te) 
            # Set v0 to thermal speed to 10 eV neutral 
            v0 = torch.sqrt(2*10*CONST.Q / (mu*CONST.H_MASS))

        elif mesh_type == 'h2':
            #Estimate total reaction rate for destruction of molecules and for interation with side walls
            # Te_torch = numpy_to_torch(Te, device, dtype)
            # n_torch = numpy_to_torch(n, device, dtype)
            react_rate = n*sigmav_ion_hh(Te) + n*sigmav_h1s_h1s_hh(Te) + n*sigmav_h1s_h2s_hh(Te)
            #directed random velocity of diatomic molecule
            v0 = torch.sqrt(8.0*CONST.TWALL*CONST.Q / (torch.pi*2*mu*CONST.H_MASS))

        else:
            raise Exception("ERROR: Mesh type invalid:", mesh_type)

        # Determine x range for atoms by finding distance into plasma where density persists.
        y = torch.zeros_like(x)
        for k in range(1, x.numel()): 
            y[k] = y[k-1] - ((x[k] - x[k-1])*0.5*(react_rate[k] + react_rate[k-1]))/v0
        if mesh_type == 'h':
            # Find x location where Y = -5, i.e. where nH should be down by exp(-5)
            xmax = torch.minimum(numpy_to_torch(interp_1d(torch_to_numpy(y), torch_to_numpy(x), -5, fill_value="extrapolate"), x.device, x.dtype), torch.max(x))
        elif mesh_type == 'h2':
            #Find x location where Y = -10, i.e., where nH2 should be down by exp(-10)
            # xmax = np.minimum(interp_1d(y, x, -10.0), max(x))
            xmax = torch.minimum(numpy_to_torch(interp_1d(torch_to_numpy(y), torch_to_numpy(x), -10), x.device, x.dtype), torch.max(x))
        xmin = x[0]


        # Interpolate Ti and Te onto a fine mesh between xmin and xmax 
        xfine = xmin + (xmax - xmin)*torch.arange(1001, dtype=x.dtype, device=x.device)/1000

        # Tifine = interp_1d(x, Ti, xfine, fill_value="extrapolate")
        # Tefine = interp_1d(x, Te, xfine)
        # nfine = interp_1d(x, n, xfine)
        # PipeDiafine = interp_1d(x, PipeDia, xfine)
        Tifine = numpy_to_torch(interp_1d(torch_to_numpy(x), torch_to_numpy(Ti), torch_to_numpy(xfine), fill_value="extrapolate"), x.device, x.dtype)
        Tefine = numpy_to_torch(interp_1d(torch_to_numpy(x), torch_to_numpy(Te), torch_to_numpy(xfine)), x.device, x.dtype)
        nfine = numpy_to_torch(interp_1d(torch_to_numpy(x), torch_to_numpy(n), torch_to_numpy(xfine)), x.device, x.dtype)
        PipeDiafine = numpy_to_torch(interp_1d(torch_to_numpy(x), torch_to_numpy(PipeDia), torch_to_numpy(xfine)), x.device, x.dtype)


        # Set up a vx, vr mesh based on raw data to get typical vx, vr values 
        vx, vr, Tnorm = self.create_vr_vx_mesh(nv, Tifine, E0=E0)

        vth = torch.sqrt( (2*CONST.Q*Tnorm) / (mu*CONST.H_MASS))
        # Estimate interaction rate with side walls
        nxfine = xfine.numel()
        gamma_wall = torch.zeros(nxfine, dtype=x.dtype, device=x.device)
        for k in range(nxfine):
            if PipeDiafine[k] > 0:
                gamma_wall[k] = 2 * max(vr) * vth / PipeDiafine[k]
        
        # Estimate total reaction rate, including charge exchange and elastic scattering, and interaction with side walls 
        if mesh_type == 'h':
            minVr = vth*torch.min(vr)
            minE0 = 0.5*CONST.H_MASS*(minVr**2) / CONST.Q

            ion_rate_option = get_config()['kinetic_h']['ion_rate']
            if ion_rate_option == 'collrad':
                ioniz_rate = collrad_sigmav_ion_h0(nfine, Tefine)
            elif ion_rate_option == 'jh':
                if (jh == None):
                    jh = Johnson_Hinnov()
                ioniz_rate = jh.jhs_coef(nfine, Tefine, no_null = True)
            else:
                ioniz_rate = sigmav_ion_h0(Tefine)
            react_rate = nfine*(ioniz_rate + sigmav_cx_h0(Tifine, torch.full_like(xfine, minE0))) + gamma_wall

        elif mesh_type == 'h2':
            react_rate = nfine*(sigmav_ion_hh(Tefine) + sigmav_h1s_h1s_hh(Tefine) + sigmav_h1s_h2s_hh(Tefine) + 0.1*sigmav_cx_hh(Tifine,Tifine)) + gamma_wall
        
        # Compute local maximum grid spacing dx_max = 2
        dx_max = torch.minimum(fctr*0.8*(2*vth*min(vr)/react_rate), torch.tensor(0.02*fctr, dtype=x.dtype, device=x.device))

        # # Construct xH Axis 
        # xpt = xmax
        # xH = np.array([xpt])
        
        # while xpt > xmin:
        #     xH = np.concatenate([np.array([xpt]), xH])
        #     dxpt1 = interp_1d(xfine, dx_max, xpt, fill_value="extrapolate")
        #     dxpt2 = np.copy(dxpt1)
        #     xpt_test = xpt - dxpt1
        #     if xpt_test > xmin:
        #         dxpt2 = interp_1d(xfine, dx_max, xpt_test, fill_value="extrapolate")

        #     if mesh_type == 'h':
        #         dxh_max = 0.0005 # JWH: 0.0015 should be sufficient for D3D because scale lengths are 2.5x larger
        #         # lowered dxh_max from 5e-4 to 4e-4; original was giving mesh size errors in kinetic_h - nh
        #         dxpt = min([dxpt1, dxpt2, dxh_max])
        #     elif mesh_type == 'h2':
        #         dxpt = min([dxpt1,dxpt2])

        #     xpt -= dxpt 
        # xH = np.concatenate([np.array([xmin]), xH[0:np.size(xH) - 1]])


        # TiH = np.interp(xH, xfine, Tifine)
        # TeH = interp_1d(xfine, Tefine, xH)
        # neH = interp_1d(xfine, nfine, xH)
        # PipeDiaH = interp_1d(xfine, PipeDiafine, xH)

        # Construct xH Axis
        xpt = xmax
        xH = torch.tensor([xpt], device=x.device, dtype=x.dtype)

        while xpt > xmin:
            xH = torch.cat([torch.tensor([xpt], device=x.device, dtype=x.dtype), xH])
            dxpt1 = torch_interp1d(torch.tensor([xpt], device=x.device, dtype=x.dtype), xfine, dx_max)
            dxpt2 = dxpt1.clone()
            xpt_test = xpt - dxpt1.item()
            if xpt_test > xmin:
                dxpt2 = torch_interp1d(torch.tensor([xpt_test], device=x.device, dtype=x.dtype), xfine, dx_max)

            if mesh_type == 'h':
                dxh_max = 0.0005
                dxpt = min([dxpt1.item(), dxpt2.item(), dxh_max])
            elif mesh_type == 'h2':
                dxpt = min([dxpt1.item(), dxpt2.item()])

            xpt -= dxpt

        xH = torch.cat([torch.tensor([xmin], device=x.device, dtype=x.dtype), xH[:-1]])

        TiH = torch_interp1d(xH, xfine, Tifine)
        TeH = torch_interp1d(xH, xfine, Tefine)
        neH = torch_interp1d(xH, xfine, nfine)
        PipeDiaH = torch_interp1d(xH, xfine, PipeDiafine)

        vx, vr, Tnorm = self.create_vr_vx_mesh(nv, TiH, E0=E0)


        self.mesh_type : str = mesh_type

        # self.x : torch.Tensor = torch.from_numpy(xH).to(dtype=dtype, device=device)
        # self.Ti : torch.Tensor = torch.from_numpy(TiH).to(dtype=dtype, device=device)
        # self.Te : torch.Tensor = torch.from_numpy(TeH).to(dtype=dtype, device=device)
        # self.ne : torch.Tensor = torch.from_numpy(neH).to(dtype=dtype, device=device)
        # self.PipeDia : torch.Tensor = torch.from_numpy(PipeDiaH).to(dtype=dtype, device=device)
        # self.vx : torch.Tensor = torch.from_numpy(vx).to(dtype=dtype, device=device)
        # self.vr : torch.Tensor = torch.from_numpy(vr).to(dtype=dtype, device=device)

        # self.Tnorm : float = torch.tensor(Tnorm, dtype=dtype, device=device)

        self.x : torch.Tensor = xH
        self.Ti : torch.Tensor = TiH
        self.Te : torch.Tensor = TeH
        self.ne : torch.Tensor = neH
        self.PipeDia : torch.Tensor = PipeDiaH
        self.vx : torch.Tensor = vx
        self.vr : torch.Tensor = vr


        self.Tnorm : torch.Tensor = Tnorm


    
    def init_lite(
            self, 
            mesh_type   : str, #'h' for kinetic_h_mesh, 'h2' for kinetic_h2_mesh
            mu          : int, 
            x           : NDArray,
            xH          : NDArray,
            Ti          : NDArray,
            Te          : NDArray, 
            n           : NDArray, 
            PipeDia     : NDArray,
            jh          : Johnson_Hinnov = None,
            E0          : NDArray = None, 
            fctr        : float   = 1.0,):

        if E0 is None:
            E0 = torch.tensor([0.0], device=x.device, dtype=x.dtype)

        #Get mesh size from config file
        nv = get_config()["kinetic_" + mesh_type]["mesh_size"]


        TiH = torch_interp1d(xH, x, Ti)
        TeH = torch_interp1d(xH, x, Te)
        neH = torch_interp1d(xH, x, n)
        PipeDiaH = torch_interp1d(xH, x, PipeDia)

        vx, vr, Tnorm = self.create_vr_vx_mesh(nv, TiH, E0=E0)


        self.mesh_type : str = mesh_type

        # self.x : torch.Tensor = torch.from_numpy(xH).to(dtype=dtype, device=device)
        # self.Ti : torch.Tensor = torch.from_numpy(TiH).to(dtype=dtype, device=device)
        # self.Te : torch.Tensor = torch.from_numpy(TeH).to(dtype=dtype, device=device)
        # self.ne : torch.Tensor = torch.from_numpy(neH).to(dtype=dtype, device=device)
        # self.PipeDia : torch.Tensor = torch.from_numpy(PipeDiaH).to(dtype=dtype, device=device)
        # self.vx : torch.Tensor = torch.from_numpy(vx).to(dtype=dtype, device=device)
        # self.vr : torch.Tensor = torch.from_numpy(vr).to(dtype=dtype, device=device)

        # self.Tnorm : float = torch.tensor(Tnorm, dtype=dtype, device=device)

        self.x : torch.Tensor = xH
        self.Ti : torch.Tensor = TiH
        self.Te : torch.Tensor = TeH
        self.ne : torch.Tensor = neH
        self.PipeDia : torch.Tensor = PipeDiaH
        self.vx : torch.Tensor = vx
        self.vr : torch.Tensor = vr


        self.Tnorm : torch.Tensor = Tnorm



    def create_vr_vx_mesh(self, nv: int, Ti: torch.tensor, E0: torch.tensor = None, Tmax: float = 0.0) -> tuple[NDArray, NDArray, float] :
        # Gwendolyn Galleher 
        '''
        Sets up optimum Vr and Vx velocity space mesh for Kinetic_Neutrals procedure 

        Parameters
        ----------
            nv : int
                number of elements desired in vr mesh
            Ti : ndarray
                ion temperature profile
            E0 : ndarray
                energy where a velocity is desired (optional)
            Tmax : float
                maximum temperature, ignore Ti above this value
                
        Returns
        -------
            vr: ndarray
                radial velocities
            vx: ndarray
                axial velocities
            Tnorm
                average of Ti
        '''

        if E0 is None:
            E0 = torch.tensor([0.0], dtype=Ti.dtype, device=Ti.device)

        # Ti = np.array(Ti) 
        # Ti = np.concatenate([Ti, E0[E0>0]])
        # if Tmax > 0:
        #     ii = np.where(Ti < Tmax)
        #     Ti = Ti[ii]
        
        # maxTi = Ti.max()
        # minTi = Ti.min()
        # Tnorm = np.nanmean(Ti)
        # vmax = 3.5
        # if (maxTi-minTi) <= (0.1*maxTi):
        #     v = (np.arange(nv+1)*vmax) / nv
        # else:
        #     g = 2*nv*np.sqrt(minTi/maxTi) / (1 - np.sqrt(minTi/maxTi))
        #     b = vmax / (nv*(nv + g))
        #     v = (g*b)*np.arange(nv+1) + b*(np.arange(nv+1)**2)

        # # Option: add velocity bins corresponding to E0     
        # v0 = 0
        # for k in range(np.size(E0)):
        #     if E0[k] > 0.0:
        #         v0 = np.sqrt(E0[k]/Tnorm)
        #         ii = np.argwhere(v > v0).T[0]
        #         if np.size(ii) > 0:
        #             v = np.concatenate([v[0:ii[0]], [v0], v[ii[0]:]])
        #         else: 
        #             v = np.concatenate([v, v0])
            
        # vr = v[1:]
        # vx = np.concatenate([-reverse(vr), vr]) 

        # return vx,vr,Tnorm

        Ti = torch.cat([Ti, E0[E0 > 0]])
        if Tmax > 0:
            Ti = Ti[Ti < Tmax]

        maxTi = Ti.max()
        minTi = Ti.min()
        Tnorm = Ti.nanmean()
        vmax = 3.5

        nv_range = torch.arange(nv+1, dtype=Ti.dtype, device=Ti.device)
        if (maxTi - minTi) <= (0.1 * maxTi):
            v = (nv_range * vmax) / nv
        else:
            g = 2 * nv * torch.sqrt(minTi / maxTi) / (1 - torch.sqrt(minTi / maxTi))
            b = vmax / (nv * (nv + g))
            v = (g*b)*nv_range + b*(nv_range**2)

        # Option: add velocity bins corresponding to E0
        v0 = 0
        for k in range(E0.numel()):
            if E0[k] > 0.0:
                v0 = torch.sqrt(E0[k] / Tnorm)
                ii = (v > v0).nonzero(as_tuple=True)[0]
                if ii.numel() > 0:
                    v = torch.cat([v[:ii[0]], v0.unsqueeze(0), v[ii[0]:]])
                else:
                    v = torch.cat([v, v0.unsqueeze(0)])

        vr = v[1:]
        vx = torch.cat([-torch.flip(vr, dims=[0]), vr])

        return vx, vr, Tnorm


    #Setup string conversion for printing
    def __str__(self):
        string = "Kinetic Mesh:\n"
        string += "    x: " + str(self.x) + "\n"
        string += "    Ti: " + str(self.Ti) + "\n"
        string += "    Te: " + str(self.Te) + "\n"
        string += "    ne: " + str(self.ne) + "\n"
        string += "    PipeDia: " + str(self.PipeDia) + "\n"
        string += "    vx: " + str(self.vx) + "\n"
        string += "    vr: " + str(self.vr) + "\n"
        string += "    Tnorm: " + str(self.Tnorm) + "\n"
        return string