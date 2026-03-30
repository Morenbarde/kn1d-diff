import torch
from torch.cuda.amp import custom_bwd, custom_fwd


# --- Torch Wrappers ---

# Source - https://stackoverflow.com/a/63964246
# Posted by jodag, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-04, License - CC BY-SA 4.0
def torch_reshape_fortran(x, shape):
    if type(shape) == int:
        # 1D output: flatten in Fortran order
        return x.t().contiguous().view(shape)
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

def numpy_to_torch(np_arr, device, dtype):
    return torch.from_numpy(np_arr).to(dtype=dtype, device=device)

def torch_to_numpy(torch_tensor):
    return torch_tensor.cpu().detach().numpy()



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


# def torch_interp1d(x, funx, funy, fill_value=None):
#     """
#     Simplified equivalent to scipy.interpolate.interp1d(funx, funy)(x)
#     Handles fill_value="extrapolate" or default (NaN at boundaries)
#     """
#     xp = funx if isinstance(funx, torch.Tensor) else torch.tensor(funx)
#     fp = funy if isinstance(funy, torch.Tensor) else torch.tensor(funy)
#     xi = x if isinstance(x, torch.Tensor) else torch.tensor(x)

#     slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
#     idx = torch.searchsorted(xp.contiguous(), xi.contiguous()) - 1

#     if fill_value == "extrapolate":
#         # Clamp to edge segments — extrapolation uses slope of first/last segment
#         idx = torch.clamp(idx, 0, len(slopes) - 1)
#     else:
#         # Default scipy behavior: NaN outside bounds
#         out_of_bounds = (idx < 0) | (idx >= len(slopes))
#         idx = torch.clamp(idx, 0, len(slopes) - 1)

#     result = fp[idx] + slopes[idx] * (xi - xp[idx])

#     if fill_value != "extrapolate":
#         result[out_of_bounds] = float('nan')

#     return result

def torch_interp1d(x : torch.Tensor, funx : torch.Tensor, funy : torch.Tensor, left=None, right=None):
    """
    Simplified equivalent to scipy.interpolate.interp1d(funx, funy, fill_value="extrapolate")(x)
    Optionally clamp left/right boundaries with fill values (like np.interp).
    """

    slopes = (funy[1:] - funy[:-1]) / (funx[1:] - funx[:-1])
    idx = torch.searchsorted(funx.contiguous(), x.contiguous()) - 1

    idx = torch.clamp(idx, 0, len(slopes) - 1)

    result = funy[idx] + slopes[idx] * (x - funx[idx])

    if left is not None:
        result = torch.where(x < funx[0], torch.full_like(result, left), result)
    if right is not None:
        result = torch.where(x > funx[-1], torch.full_like(result, right), result)

    return result


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



# --- BS2DR ---

def de_boor_basis(x: torch.Tensor, knots: torch.Tensor, order: int) -> torch.Tensor:
    n = x.shape[0]
    m = knots.shape[0]

    # Order-0 basis
    B = ((x[:, None] >= knots[None, :-1]) & (x[:, None] < knots[None, 1:])).to(x.dtype)

    # Clamp last point into final interval
    at_end = (x == knots[-1])
    if at_end.any():
        B[at_end] = 0.0
        B[at_end, -1] = 1.0

    # De Boor recursion
    for k in range(1, order + 1):
        n_basis = m - k - 1  # number of basis functions at this level

        denom_left  = knots[k:m-1]   - knots[:m-k-1]   # (n_basis,)
        denom_right = knots[k+1:m]   - knots[1:m-k]    # (n_basis,)

        safe_left  = torch.where(denom_left  != 0, denom_left,  torch.ones_like(denom_left))
        safe_right = torch.where(denom_right != 0, denom_right, torch.ones_like(denom_right))

        alpha = torch.where(
            denom_left[None, :] != 0,
            (x[:, None] - knots[None, :m-k-1]) / safe_left[None, :],
            torch.zeros(n, n_basis, dtype=x.dtype, device=x.device)
        )
        beta = torch.where(
            denom_right[None, :] != 0,
            (knots[None, k+1:m] - x[:, None]) / safe_right[None, :],
            torch.zeros(n, n_basis, dtype=x.dtype, device=x.device)
        )

        B = alpha * B[:, :n_basis] + beta * B[:, 1:n_basis+1]

    return B  # (n, m - order - 1)


def bs2dr_diff(x: torch.Tensor, y: torch.Tensor, 
          kx_ord: int, ky_ord: int,
          xknot: torch.Tensor, yknot: torch.Tensor, 
          bscoef: torch.Tensor) -> torch.Tensor:
    '''
    Differentiable bivariate B-spline evaluation, equivalent to IDL bs2dr / scipy bispeu.
    Gradients flow through x and y. xknot, yknot, bscoef are treated as fixed constants.

    Parameters
    ----------
        x, y    : (n,) query coordinates, must have requires_grad if gradient needed
        kx_ord  : spline order in x (IDL convention: degree + 1, so cubic = 4)
        ky_ord  : spline order in y
        xknot   : (mx,) knot vector in x
        yknot   : (my,) knot vector in y
        bscoef  : (nx_basis * ny_basis,) flattened spline coefficients, scipy/FITPACK ordering

    Returns
    -------
        result : (n,) evaluated spline values, differentiable w.r.t. x and y
    '''
    # Note: bispeu uses (ky_ord-1, kx_ord-1) and swapped knot order — match that here
    Bx = de_boor_basis(x, xknot, kx_ord - 1)   # (n, nx_basis)
    By = de_boor_basis(y, yknot, ky_ord - 1)   # (n, ny_basis)

    nx_basis = Bx.shape[1]
    ny_basis = By.shape[1]

    # bscoef is flattened in FITPACK order: (ny_basis, nx_basis) -> reshape accordingly
    C = bscoef.reshape(ny_basis, nx_basis)      # (ny_basis, nx_basis)

    # For each query point: result[i] = By[i] @ C @ Bx[i]
    # Vectorized: (n, ny) @ (ny, nx) -> (n, nx), then elementwise with (n, nx) -> sum -> (n,)
    result = torch.sum((By @ C) * Bx, dim=1)   # (n,)

    return result



def torch_locate(table, value):
    value = value.reshape(-1) if value.dim() > 0 else value.unsqueeze(0)
    indices = torch.searchsorted(table.contiguous(), value.contiguous()) - 1
    indices = indices.clamp(-1, len(table) - 1)
    return indices