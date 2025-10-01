# Utility Functions for KN1DPy

import json
from typing import Any

from numpy.typing import NDArray
import numpy as np
from scipy import interpolate

# Json Files

def get_json(file_path:str) -> dict[str, Any]:
    # Load json file
    with open(file_path, 'r') as config:
        return json.load(config)
    
def get_config() -> dict[str, Any]:
    # Lazy function to load config file
    return get_json('config.json')


# Printing

def debrief(statement: str, condition: bool):
    # Print statement if condition is true
    if condition:
        print(statement)


#Interpolation

def interp_1d(funx: NDArray, funy: NDArray, x: NDArray, kind: str = 'linear', axis: int = -1,
        copy: bool = True, bounds_error: Any | None = None, fill_value: float = np.nan, assume_sorted: bool = False):

    #Wrapper function for creating a scipy 1d interpolation function and run it on an array    
    interpfunc = interpolate.interp1d(funx, funy, kind=kind, axis=axis, copy=copy, bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted)
    return interpfunc(x)


#Reverse Function from reverse.pro

def reverse(a, subscript=1):
    #reverses the order of a list at the given dimension (subscript)
    #initially assume at least 1 dimension
    ndims = 1
    b = a

    #if the 1st variable is also a list then a dimension is added, recurring until no longer true
    while type(b[0]) == list:
        ndims += 1
        if len(b) == 0:
            break
        b = b[0]
    if subscript > ndims:
        raise Exception('Subscript_index must be less than or equal to number of dimensions.')
    if subscript == 1: #unique case where it is reversing the 1st dim
        a = a[::-1]
        return a
    return rev_rec(a, subscript, 1)
    
def rev_rec(a, subscript, dim_tracker):
    #a recursive function that iterates over everything in a, and reverses everything in the specified dim
    i = 0
    while i < len(a):
        if dim_tracker == subscript-1:
            a[i] = a[i][::-1]
        else:
            a[i] = rev_rec(a[i], subscript, dim_tracker+1)
        i += 1
    return a