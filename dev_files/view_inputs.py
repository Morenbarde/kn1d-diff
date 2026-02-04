from scipy.io import readsav
import numpy as np
import sys
import matplotlib.pyplot as plt
import os


def generate_plot(dir, title, x, y, xlabel, ylabel, x_lim, x_sep):
    plt.plot(x, y, color = 'teal', marker='x', markersize=0.25, markeredgecolor='red', label="KN1DPy")
    plt.axvline(x=x_lim, color='black', linestyle='--', linewidth=0.5, label="x_lim")
    plt.axvline(x=x_sep, color='purple', linestyle='--', linewidth=0.5, label="x_sep")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(dir + title + '.png', dpi=300)
    plt.clf()



np.set_printoptions(linewidth=225)
np.set_printoptions(threshold=sys.maxsize)

standard_out = sys.stdout

##Input

data_file = './sav_files/kn1d_test_inputs.sav'
data_file = './sav_files/1090904018_950to1050.sav'
# data_file = './sav_files/1090904029_950to1050_towall.sav'
print("Loading file: "  + data_file)
sav_data = readsav(data_file)

keys = ['x', 'x_lim', 'x_sep', 'p_wall', 'mu', 't_i', 't_e', 'n_e', 'vx', 'lc', 'd_pipe']
exclude_plotting = ['x', 'x_lim', 'x_sep', 'p_wall', 'mu']


# Generate Plotting Dir if nonexistant

plot_dir = 'Inputs/Plots/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

x = sav_data['x'] # x axis for plotting

# Print and plot
for key in keys:
    data = np.asarray(sav_data[key])
    print(f"{key}:")
    print(f"Length: {data.size}")
    print(np.asarray(data))
    print()

    if key not in exclude_plotting:
        generate_plot(plot_dir, key, x, data, 'x (m)', key, sav_data['x_lim'], sav_data['x_sep'])

