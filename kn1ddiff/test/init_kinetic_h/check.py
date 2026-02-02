import torch
import numpy as np

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.kinetic_h import KineticH

dir = "kn1ddiff/test/init_kinetic_h/"

torch.set_default_dtype(torch.float64)

mesh_input = np.load(dir+"h_mesh_in.npz")
kh_in = np.load(dir+"kinetic_h_in.npz")
kh_par = np.load(dir+"kinetic_h_params.npz")

print(kh_par["vth"])


mesh1 = KineticMesh('h', mesh_input["mu"], mesh_input["x"], mesh_input["Ti"], mesh_input["Te"], mesh_input["n"], mesh_input["PipeDia"], E0=mesh_input["E0"], fctr=mesh_input["fctr"], param_type='torch')
kinetic_h = KineticH(mesh1, torch.from_numpy(kh_in["mu"]), torch.from_numpy(kh_in["vxiA"]), torch.from_numpy(kh_in["fHBC"]), torch.from_numpy(kh_in["GammaxHBC"]), 
                     ni_correct=True, truncate=1.0e-3, max_gen=100, 
                     compute_errors=True, debrief=True, debug=False)

for key, value in kh_par.items():
    print("Checking "+key)
    print("Type: ", type(getattr(kinetic_h, key)))
    print("Param is close:", np.allclose(value, getattr(kinetic_h, key)))