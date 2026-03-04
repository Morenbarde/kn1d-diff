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