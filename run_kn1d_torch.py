from dataclasses import dataclass, asdict

from kn1ddiff.kn1d import kn1d, KN1DResults
from scipy.io import readsav
import numpy as np
import torch
import sys

import time

import sys
np.set_printoptions(linewidth=225)
np.set_printoptions(threshold=sys.maxsize)

standard_out = sys.stdout

# Torch
dtype = torch.float64
USE_CPU = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda and not USE_CPU else "cpu")
print("device: ", device)


##Input

data_file = './sav_files/kn1d_test_inputs.sav'
# data_file = './sav_files/1090904018_950to1050.sav'
# data_file = './sav_files/1090904029_950to1050_towall.sav'
print("Loading file: "  + data_file)
sav_data = readsav(data_file)

for key, value in sav_data.items():
    sav_data[key] = torch.tensor(np.asarray(value).astype(np.float32), dtype=dtype, device=device)

##Output

print("Beginning KN1D")
start = time.time()
results = kn1d(sav_data['x'], sav_data['x_lim'], sav_data['x_sep'], sav_data['p_wall'], sav_data['mu'], sav_data['t_i'], 
               sav_data['t_e'], sav_data['n_e'], sav_data['vx'], sav_data['lc'], sav_data['d_pipe'],
               xH=None, xH2=None, 
               max_gen=100, Hdebug=0, H2debug=0, debrief = 1, Hdebrief = 1, H2debrief = 1, compute_errors = 1)
end = time.time()


print("Elapsed Time: ", end-start)
print()

#print result data
output = open('Results/torch_output.txt', 'w')
sys.stdout = output

for key, value in asdict(results).items():
    print(key)
    print(value)
    print()

output.close()
sys.stdout = standard_out