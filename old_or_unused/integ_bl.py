import numpy as np

from .utils import reverse

#NOTE Replace with scipy.integrate.cumulative_trapeziod?

def integ_bl(x, y, value_only=None, rev=None):
    # X, Y are input np.arrays
    # value_only returns only the final calculated value if set.
    #   -default returns list
    # unsure what rev is used for, but if set it reverses X, Y, and the output
    """
        Constructs velocity space differentials for distribution functions.

        Parameters:
        -----------
        x : np.ndarray
            Array of radial velocities.
        y : np.ndarray
            Array of axial velocities.

        returns:
        --------
        result : 
    """

    if rev != None: #reverses arrays if specified
        y = reverse(y)
        x = -reverse(x)

    result = np.zeros(y.size)
    for i in range(1,y.size):
        result[i] = result[i-1] + 0.5*(x[i] - x[i-1])*(y[i] + y[i-1])

    if value_only != None: #returns only last element
        return result[-1]
    if rev != None: #reverses output
        return reverse(result)
    return result
