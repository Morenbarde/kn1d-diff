import torch
from torch.cuda.amp import custom_bwd, custom_fwd


# class DifferentiableClamp(torch.autograd.Function):
#     """
#     https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
#     In the forward pass this operation behaves like torch.clamp.
#     But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
#     """

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, input, min, max):
#         return input.clamp(min=min, max=max)

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, grad_output):
#         return grad_output.clone(), None, None


# def dclamp(input, min, max):
#     """
#     Like torch.clamp, but with a constant 1-gradient.
#     :param input: The input that is to be clamped.
#     :param min: The minimum value of the output.
#     :param max: The maximum value of the output.
#     """
#     return DifferentiableClamp.apply(input, min, max)


def dclamp(x, min=None, max=None):
    clamped = torch.clamp(x, min=min, max=max)
    return x + (clamped - x).detach()


def poly_torch(x: torch.Tensor, c: torch.Tensor):
    '''
    Evaluate a polynomial at one or more points, but with pytorch

    Parameters
    ----------
        x : float or ndarray
            Variable/s to evaluate the polynomial at
        c : ndarray
            array of polynomial coefficients
            
    Returns
    -------
        y : float or ndarray
            Value of the polynomial evaluated at x, array of values if x is an array
    '''

    n = c.numel()-1
    y = c[n]
    for i in range(n-1, -1, -1):
        y = y*x + c[i]
    return y



# Interpolation

import torch
import torch.nn.functional as F

def bilinear_interp_rectgrid(logsigmav, indne, indte):
    """
    Gradient-safe replacement for RectBivariateSpline(kx=1, ky=1) + diagonal.
    
    Args:
        logsigmav: (xs, ys) tensor — your log-sigma grid
        indne:     (N,) tensor — row indices (float, can be fractional)
        indte:     (N,) tensor — col indices (float, can be fractional)
    
    Returns:
        (N,) tensor — exp of interpolated log-sigma values (i.e. sigma)
    """
    xs, ys = logsigmav.shape

    # grid_sample expects input shape (N, C, H, W)
    grid_input = logsigmav[None, None, :, :]  # (1, 1, xs, ys)

    # Normalize indices to [-1, 1] as required by grid_sample
    # grid_sample treats dim -1 as x (columns) and dim -2 as y (rows)
    norm_x = (indte / (ys - 1)) * 2 - 1  # column indices → x
    norm_y = (indne / (xs - 1)) * 2 - 1  # row indices    → y

    # Build sampling grid: shape (1, N, 1, 2)
    grid = torch.stack([norm_x, norm_y], dim=-1)[None, :, None, :]

    # Sample — output shape (1, 1, N, 1) → squeeze to (N,)
    sampled = F.grid_sample(
        grid_input, grid,
        mode='bilinear',
        padding_mode='border',   # matches spline edge behaviour
        align_corners=True       # index 0 → -1, index max → +1
    )
    sampled = sampled[0, 0, :, 0]  # (N,)

    return torch.exp(sampled)


# def path_interp_2d_torch(p, px, py, x, y):
#     """
#     Gradient-safe bilinear interpolation on a regular 2D grid.
    
#     Args:
#         p:  (H, W) tensor — the values on the grid
#         px: (H,)   tensor — grid coordinates along dim 0
#         py: (W,)   tensor — grid coordinates along dim 1
#         x:  (N,)   tensor — query points along dim 0
#         y:  (N,)   tensor — query points along dim 1
    
#     Returns:
#         (N,) tensor of interpolated values
#     """
#     # Normalize query points to [-1, 1] as required by grid_sample
#     x_norm = 2.0 * (x - px[0]) / (px[-1] - px[0]) - 1.0
#     y_norm = 2.0 * (y - py[0]) / (py[-1] - py[0]) - 1.0

#     # grid_sample expects (N, C, H, W) input and (N, H_out, W_out, 2) grid
#     p_4d = p.unsqueeze(0).unsqueeze(0)                          # (1, 1, H, W)
#     grid = torch.stack([x_norm, y_norm], dim=-1)                # (N, 2) — note xy order
#     grid = grid.unsqueeze(0).unsqueeze(0)                       # (1, 1, N, 2)

#     result = F.grid_sample(p_4d, grid, mode='bilinear', 
#                            align_corners=True, padding_mode='border')
#     return result.squeeze()                                      # (N,)


def path_interp_2d_torch(p, px, py, x, y):
    """
    Gradient-safe bilinear interpolation on a regular 2D grid.
    Matches scipy RegularGridInterpolator((px, py), p) exactly.

    Args:
        p:  (Nx, Ny) tensor — grid values where Nx=len(px), Ny=len(py)
        px: (Nx,)    tensor — grid coords for axis 0 (rows)
        py: (Ny,)    tensor — grid coords for axis 1 (cols)
        x:  (N,)     tensor — query coords along axis 0
        y:  (N,)     tensor — query coords along axis 1

    Returns:
        (N,) tensor of interpolated values
    """
    # --- axis 0 (px) ---
    # Find lower bin index for each query point
    ix = torch.searchsorted(px.contiguous(), x.contiguous(), right=True) - 1
    ix = torch.clamp(ix, 0, len(px) - 2)

    # Compute interpolation weight along axis 0
    x0 = px[ix]
    x1 = px[ix + 1]
    tx = (x - x0) / (x1 - x0)          # in [0,1]

    # --- axis 1 (py) ---
    iy = torch.searchsorted(py.contiguous(), y.contiguous(), right=True) - 1
    iy = torch.clamp(iy, 0, len(py) - 2)

    y0 = py[iy]
    y1 = py[iy + 1]
    ty = (y - y0) / (y1 - y0)          # in [0,1]

    # --- Bilinear interpolation from the four corners ---
    p00 = p[ix,     iy    ]
    p10 = p[ix + 1, iy    ]
    p01 = p[ix,     iy + 1]
    p11 = p[ix + 1, iy + 1]

    return (p00 * (1 - tx) * (1 - ty) +
            p10 *      tx  * (1 - ty) +
            p01 * (1 - tx) *      ty  +
            p11 *      tx  *      ty)