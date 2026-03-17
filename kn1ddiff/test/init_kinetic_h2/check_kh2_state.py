import torch
import numpy as np
import json

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.kinetic_h2 import KineticH2
from kn1ddiff.test.utils import rel_L2_np, rel_L2_torch

dir = "kn1ddiff/test/init_kinetic_h2/"

dtype = torch.float64


if __name__ == "__main__":

    use_cuda = False #torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)

    with open(dir+"h2_mesh_in.json", 'r') as f:
        mesh_input = json.load(f)
    with open(dir+"h2_mesh_out.json", 'r') as f:
        mesh_output = json.load(f)
    with open(dir+"kinetic_h2_in.json", 'r') as f:
        kh2_in = json.load(f)
    with open(dir+"kinetic_h2_params.json", 'r') as f:
        kh2_param = json.load(f)
    with open(dir+"kinetic_h2_internal1.json", 'r') as f:
        kh2_internal1 = json.load(f)
    with open(dir+"kinetic_h2_internal2.json", 'r') as f:
        kh2_internal2 = json.load(f)

    # Convert Mesh input to numpy
    for key, value in mesh_input.items():
        mesh_input[key] = np.asarray(value)
    # Convert Mesh output to torch
    for key, value in mesh_output.items():
        mesh_output[key] = torch.tensor(value, dtype=dtype, device=device)

    # Convert KH input and params to tensor
    for key, value in kh2_in.items():
        kh2_in[key] = torch.tensor(value, dtype=dtype, device=device)
    for key, value in kh2_param.items():
        kh2_param[key] = torch.tensor(value, dtype=dtype, device=device)
    kh2_internal = kh2_internal1 | kh2_internal2
    for key, value in kh2_internal.items():
        if kh2_internal[key] is not None:
            kh2_internal[key] = torch.tensor(value, dtype=dtype, device=device)


    # Generate Mesh and initializat kh
    mesh = KineticMesh('h2', mesh_input["mu"], mesh_input["x"], mesh_input["Ti"], mesh_input["Te"], mesh_input["n"], mesh_input["PipeDia"], E0=mesh_input["E0"], fctr=mesh_input["fctr"], device=device, dtype=dtype)
    
    # print("Checking Mesh")
    # for key, value in mesh_output.items():
    #     print("Checking "+key)
    #     attr = getattr(mesh, key)

    #     result = torch.allclose(value, attr)
    #     print("                 L2: ", rel_L2_torch(value, attr))
    #     print("                 Param is close:", result)
    # input()

    kinetic_h2 = KineticH2(mesh, kh2_in["mu"], kh2_in["vxi"], kh2_in["fH2BC"], kh2_in["GammaxH2BC"], kh2_in["NuLoss"], kh2_in["SH2_initial"],
                        compute_h_source=True, ni_correct=True, truncate=1.0e-3, max_gen=100, 
                        compute_errors=True, debrief=True, debug=False, 
                        device=device, dtype=dtype)

    # Check main parameters
    for key, value in kh2_param.items():
        print("Checking "+key)

        attr = getattr(kinetic_h2, key)
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
    for key, value in kh2_internal.items():
        print("Checking Internal "+key)

        attr = getattr(kinetic_h2.Internal, key)

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
        # if result == False:
        #     print("True", value)
        #     print("Computed", attr)
