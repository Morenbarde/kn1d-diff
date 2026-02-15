import torch
import numpy as np
import json

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.test.utils import numpy_to_torch, rel_L2_torch, rel_L2_np
from kn1ddiff.make_dvr_dvx import VSpace_Differentials

from KN1DPy.make_dvr_dvx import VSpace_Differentials as OG_DIFF

directory = "kn1ddiff/test/mesh_torch/"
mesh_type = 'h'

in_data_file = mesh_type+"_mesh_in.json"
out_data_file = mesh_type+"_mesh_out.json"

dtype = torch.float64


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)


    with open(directory+in_data_file, 'r') as f:
        in_data = json.load(f)
    with open(directory+out_data_file, 'r') as f:
        out_data = json.load(f)

    for key, value in in_data.items():
        in_data[key] = np.asarray(value)

    out_data_np = {}
    for key, value in out_data.items():
        out_data[key] = torch.tensor(value, device=device, dtype=dtype)
        out_data_np[key] = np.asarray(value)


    print("Checking Mesh")

    mesh = KineticMesh(mesh_type, in_data["mu"], in_data["x"], in_data["Ti"], in_data["Te"], in_data["n"], in_data["PipeDia"], E0=in_data["E0"], fctr=in_data["fctr"], device=device, dtype=dtype)

    for key, value in out_data.items():
        print("Checking "+key+":")
        print(torch.allclose(value, getattr(mesh, key)))
        print("L2: ", rel_L2_torch(value, getattr(mesh, key)))

    print()


    print("Checking VSpace Differentials")

    original_diff = OG_DIFF(out_data_np["vr"], out_data_np["vx"])
    torch_diff = VSpace_Differentials(mesh.vr, mesh.vx)

    for attr in [name for name in dir(torch_diff) if not name.startswith("__")]:
        print("Checking "+attr)
        print(np.allclose(getattr(original_diff, attr), getattr(torch_diff, attr).cpu()))
        print("L2: ", rel_L2_np(getattr(original_diff, attr), getattr(torch_diff, attr).cpu().numpy()))


