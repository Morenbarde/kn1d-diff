import numpy as np

def poly(x, c):
    '''
    Evaluate a polynomial at one or more points

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

    # NOTE Clean/Remove Type Checking, currently messing with test script
    if type(x) == list:
        x = np.array(x)
    n = len(c)-1
    y = c[n]
    for i in range(n-1, -1, -1):
        y = y*x + c[i]
    return y
