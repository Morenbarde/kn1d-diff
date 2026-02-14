import numpy as np
from numpy.typing import NDArray
import torch

''' 
Authors: Julio Balbin, Carlo Becerra
Date: August 17th, 2024
'''
# Restructured into class - Nicholas Brown, Oct 7, 2025

class VSpace_Differentials:
    '''
    Constructs velocity space differentials for distribution functions

    Attributes
    ----------
        dvr_vol : ndarray
            Differential volume element for radial velocities.
        dvr_vol_h_order : ndarray
            Differential volume element for radial velocities (higher order).
        dvx : ndarray
            Differential for axial velocities.
        dvr : ndarray
            Differential for radial velocities.
        vr_left_bound, vr_right_bound : ndarray
            Left and right boundaries for radial velocities.
        vx_left_bound, vx_right_bound : ndarray
            Left and right boundaries for axial velocities.
        volume : ndarray
            Volume elements in velocity space.
        vth_dvx, vx_dvx, vr_dvr : ndarray
            Auxiliary quantities for kinetic equations.
        vmag_squared : ndarray
            Squared magnitude of the velocity.
        vx_fpi, vx_lpi, vx_fni, vx_lni : int
            Indices for positive and negative axial velocities.
    '''
    
    def __init__(self, vr : torch.Tensor, vx : torch.Tensor):

        '''
        Parameters
        -----------
            vr : ndarray
                Array of radial velocities.
            vx : ndarray
                Array of axial velocities.
        '''

        dtype = vr.dtype
        device = vr.device

        nvr = vr.numel()
        nvx = vx.numel()

        # --- Calculations for r-dimension ---
        # vr_extend = np.append(vr, 2*vr[-1] - vr[-2])
        vr_extend = torch.cat([vr, (2*vr[-1] - vr[-2]).unsqueeze(0)])
        # vr_mid = np.concatenate(([0.0], 0.5*(vr_extend + np.roll(vr_extend, -1)))) #midpoints between each value in vr
        vr_mid = torch.cat([
                torch.zeros(1, dtype=dtype, device=device), 
                 0.5*(vr_extend + torch.roll(vr_extend, -1))
        ]) #midpoints between each value in vr

        self.vr_right_bound = vr_mid[1:nvr+1]
        # self.vr_left_bound = np.copy(vr_mid[0:nvr])
        self.vr_left_bound = torch.clone(vr_mid[0:nvr])
        self.dvr = self.vr_right_bound - self.vr_left_bound

        self.dvr_vol = np.pi * (self.vr_right_bound**2 - self.vr_left_bound**2)
        self.dvr_vol_h_order = (4/3)*np.pi*(self.vr_right_bound**3 - self.vr_left_bound**3)


        # --- Calculations for x-dimension ---
        # vx_extend = np.concatenate(([2*vx[0] - vx[1]], vx, [2*vx[-1] - vx[-2]]))
        vx_extend = torch.concatenate([
            (2*vx[0] - vx[1]).unsqueeze(0), 
            vx, 
            (2*vx[-1] - vx[-2]).unsqueeze(0)
        ])

        self.vx_right_bound = 0.5*(torch.roll(vx_extend, -1) + vx_extend)[1:nvx+1]
        self.vx_left_bound =  0.5*(torch.roll(vx_extend,  1) + vx_extend)[1:nvx+1]
        self.dvx = self.vx_right_bound - self.vx_left_bound


        # --- volume calculation ---
        self.volume = self.dvr_vol[:, None] * self.dvx


        # --- compute velocites over differentials ---
        self.vth_dvx = torch.zeros((nvr+2,nvx+2), dtype=dtype, device=device)
        self.vx_dvx  = torch.zeros((nvr+2,nvx+2), dtype=dtype, device=device)
        self.vr_dvr  = torch.zeros((nvr+2,nvx+2), dtype=dtype, device=device)
        self.vth_dvx[1:nvr+1, 1:nvx+1] = 1.0 / self.dvx
        self.vx_dvx[ 1:nvr+1, 1:nvx+1] = vx  / self.dvx
        self.vr_dvr[ 1:nvr+1, 1:nvx+1] = vr[:, None] / self.dvr[:, None]


        # --- Compute velocity magnitude squared ---
        self.vmag_squared = vr[:, None]**2 + vx**2 


        # --- Get positive and negaitve indices from vx
        # pos_vx_indices = np.where(vx>0)[0]
        pos_vx_indices = torch.where(vx > 0)[0]
        self.vx_pos_start = int(pos_vx_indices[0])  # First Positive Index
        self.vx_pos_end = int(pos_vx_indices[-1]) # Last Positive Index
 
        # neg_vx_indices = np.where(vx<0)[0]
        neg_vx_indices = torch.where(vx<0)[0]
        self.vx_neg_start = int(neg_vx_indices[0])  # First Negative Index
        self.vx_neg_end = int(neg_vx_indices[-1]) # Last Negative Index

    
    #Setup string conversion for printing
    def __str__(self):
        string = "Velocity Space Differentials:\n"
        string += "    dvr_vol: " + str(self.dvr_vol) + "\n"
        string += "    dvr_vol_h_order: " + str(self.dvr_vol_h_order) + "\n"
        string += "    dvx: " + str(self.dvx) + "\n"
        string += "    dvr: " + str(self.dvr) + "\n"
        string += "    vr_right_bound: " + str(self.vr_right_bound) + "\n"
        string += "    vr_left_bound: " + str(self.vr_left_bound) + "\n"
        string += "    vx_right_bound: " + str(self.vx_right_bound) + "\n"
        string += "    vx_left_bound: " + str(self.vx_left_bound) + "\n"
        string += "    volume: " + str(self.volume) + "\n"
        string += "    vth_dvx: " + str(self.vth_dvx) + "\n"
        string += "    vx_dvx: " + str(self.vx_dvx) + "\n"
        string += "    vr_dvr: " + str(self.vr_dvr) + "\n"
        string += "    vmag_squared: " + str(self.vmag_squared) + "\n"
        string += "    pos index range: " + str(self.pos_vx0) + ", " + str(self.pos_vxn) + "\n"
        string += "    vx_right_bound: " + str(self.neg_vx0) + ", " + str(self.neg_vxn)  + "\n"
        return string