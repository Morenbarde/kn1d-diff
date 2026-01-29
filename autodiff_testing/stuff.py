import torch
import numpy as np

from test_functions import *



# x = torch.tensor([124.2, 153.2, 164.3])
# y = torch.tensor([1200.3, 1500.3, 1600.3])

# result1, result2, result3 = mult_var(x, y)

# print(result1.detach().cpu().numpy())
# print(result2.detach().cpu().numpy())
# print(result3.detach().cpu().numpy())


v0 = torch.tensor([2000.0, 3100.3, 1842.3, 1902.7, 1932.96])          # m/s
theta = torch.tensor([0.7, 0.2, 0.4, 0.6, 0.37])        # radians
m = torch.tensor([10.5, 40.2, 41.5, 24.1, 9.0])             # kg
c = torch.tensor([0.1, 0.32, 0.52, 0.23, 0.08]) 

x, y = simulate_projectile(v0, theta, m, c)

print(x.detach().cpu().numpy())
print(y.detach().cpu().numpy())