# Utility Functions for KN1DPy

import json
from typing import Any

from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.io import readsav
import netCDF4 as nc

# --- Json Files ---

def get_json(file_path:str) -> dict[str, Any]:
    # Load json file
    with open(file_path, 'r') as config:
        return json.load(config)
    
def get_config() -> dict[str, Any]:
    # Lazy function to load config file
    return get_json('config.json')


# --- Printing ---

def debrief(statement: str, condition: bool):
    # Print statement if condition is true
    if condition:
        print(statement)

def sval(s,length=None):
  # removes leading / trailing spaces and truncates string to a specified length
  return str(s).strip()[:length]


# --- Interpolation ---

def interp_1d(funx: NDArray, funy: NDArray, x: NDArray, kind: str = 'linear', axis: int = -1,
        copy: bool = True, bounds_error: Any | None = None, fill_value: float = np.nan, assume_sorted: bool = False):

    #Wrapper function for creating a scipy 1d interpolation function and run it on an array    
    interpfunc = interp1d(funx, funy, kind=kind, axis=axis, copy=copy, bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted)
    return interpfunc(x)

def path_interp_2d(p, px, py, x, y):
    interp = RegularGridInterpolator((px, py), p, method='linear')
    points = np.column_stack([x, y])
    return interp(points)


# --- Reverse Function from reverse.pro ---

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


# --- Read Functions ---

def sav_read(sav_path, nc_path):
    # used to read and save .sav files

    # Inputs:
    #   sav_path - the path to the .sav input file
    #   nc_path  - the path to the .nc file you are creating 
    #Ouputs:
    #    input_dict - a dictionary of all inputs from the input file

    sav_data = readsav(sav_path)
    fn = nc_path
    ds = nc.Dataset(fn, 'w', format = 'NETCDF4') 
    for k,v in sav_data.items(): 
        setattr(ds, k, v)
    input_dict = ds.__dict__
    return input_dict

def nc_read(nc_path):
    # Used to read and save .nc files (netCDF)

    # Inputs:
    #   nc_path  - the path to the .nc file you are creating 
    # Ouputs:
    #    input_dict - a dictionary of all inputs from the input file
    
    fn = nc_path
    ds = nc.Dataset(fn) 
    input_dict = ds.__dict__
    return input_dict