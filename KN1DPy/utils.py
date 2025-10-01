# Shared function definition to simplify print statements done under the debrief options

import json
from typing import Any

#Load json file
def get_config(file_path:str) -> dict[str, Any]:
    with open(file_path, 'r') as config:
        return json.load(config)

# Print statement if condition is true
def debrief(statement, condition):
    if condition:
        print(statement)


#Reverse Function from reverse.pro
#reverses the order of a list at the given dimension (subscript)
def reverse (a, subscript=1):
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
    
#a recursive function that iterates over everything in a, and reverses everything in the specified dim
def rev_rec(a, subscript, dim_tracker):
    i = 0
    while i < len(a):
        if dim_tracker == subscript-1:
            a[i] = a[i][::-1]
        else:
            a[i] = rev_rec(a[i], subscript, dim_tracker+1)
        i += 1
    return a