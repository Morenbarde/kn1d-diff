import torch
import numpy as np
import json

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.kinetic_h import KineticH
from kn1ddiff.test.utils import rel_L2_np, rel_L2_torch

dir = "kn1ddiff/test/init_kinetic_h/"

dtype = torch.float64


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)

    with open(dir+"h_mesh_in.json", 'r') as f:
        mesh_input = json.load(f)
    with open(dir+"kinetic_h_in.json", 'r') as f:
        kh_in = json.load(f)
    with open(dir+"kinetic_h_params.json", 'r') as f:
        kh_param = json.load(f)
    with open(dir+"kinetic_h_internal.json", 'r') as f:
        kh_internal = json.load(f)

    # Convert Mesh input to numpy
    for key, value in mesh_input.items():
        mesh_input[key] = np.asarray(value)

    # Convert KH input and params to tensor
    for key, value in kh_in.items():
        kh_in[key] = torch.tensor(value, dtype=dtype, device=device)
    for key, value in kh_param.items():
        kh_param[key] = torch.tensor(value, dtype=dtype, device=device)
    for key, value in kh_internal.items():
        if kh_internal[key] is not None:
            kh_internal[key] = torch.tensor(value, dtype=dtype, device=device)


    # Generate Mesh and initializat kh
    mesh = KineticMesh('h', mesh_input["mu"], mesh_input["x"], mesh_input["Ti"], mesh_input["Te"], mesh_input["n"], mesh_input["PipeDia"], E0=mesh_input["E0"], fctr=mesh_input["fctr"], device=device, dtype=dtype)
    
    kinetic_h = KineticH(mesh, kh_in["mu"], kh_in["vxi"], kh_in["fHBC"], kh_in["GammaxHBC"], 
                        ni_correct=True, truncate=1.0e-3, max_gen=100, 
                        compute_errors=True, debrief=True, debug=False, 
                        device=device, dtype=dtype)

    # Check main parameters
    for key, value in kh_param.items():
        print("Checking "+key)

        attr = getattr(kinetic_h, key)
        if type(attr) != torch.Tensor:
            attr = torch.tensor(attr, dtype=dtype, device=device)
        elif attr.dtype != dtype:
            attr = attr.to(dtype=dtype)

        result = torch.allclose(value, attr)
        print("                 L2: ", rel_L2_torch(value, attr))
        print("                 Param is close:", result)

    print()
    print()

    # Check Internal Block parameters
    for key, value in kh_internal.items():
        print("Checking Internal "+key)

        attr = getattr(kinetic_h.Internal, key)

        if value is None or attr is None:
            result = attr is value
        else:
            if type(attr) != torch.Tensor:
                attr = torch.tensor(attr, dtype=dtype, device=device)
            elif attr.dtype != dtype:
                attr.to(dtype=dtype)
            result = torch.allclose(value, attr)
            print("                 L2: ", rel_L2_torch(value, attr))

        print("                 Param is close:", result)
        if result == False:
            print("True", value)
            print("Computed", attr)
