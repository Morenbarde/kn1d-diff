import torch
import numpy as np

from kn1ddiff.kinetic_mesh import *

dir = "kn1ddiff/test/mesh_torch/"
mesh_type = 'h'

in_data_file = mesh_type+"_mesh_in.npz"
out_data_file = mesh_type+"_mesh_out.npz"



in_data = np.load(dir+in_data_file)
out_data = np.load(dir+out_data_file)


mesh1 = KineticMesh(mesh_type, in_data["mu"], in_data["x"], in_data["Ti"], in_data["Te"], in_data["n"], in_data["PipeDia"], E0=in_data["E0"], fctr=in_data["fctr"])
mesh2 = KineticMesh(mesh_type, in_data["mu"], in_data["x"], in_data["Ti"], in_data["Te"], in_data["n"], in_data["PipeDia"], E0=in_data["E0"], fctr=in_data["fctr"], param_type='torch')

for key, value in out_data.items():
    print("Checking "+key)
    print("Numpy is close:", np.allclose(value, getattr(mesh1, key)))
    print("Torch is close:", np.allclose(value, getattr(mesh2, key).numpy()))

