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