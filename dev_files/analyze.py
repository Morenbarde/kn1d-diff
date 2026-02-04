import numpy as np
from scipy.io import readsav
import matplotlib.pyplot as plt
import sys

plot_output = 'Results/Plots/'

def perror(a, b):
    return np.abs((a - b)/a)

def mean_perror(a, b):
    return np.mean(perror(a, b))

def max_perror(a, b):
    return np.max(perror(a, b))

def generate_compare_plot(title, x, y, idl_x, idl_y, xlabel, ylabel):
    plt.plot(x, y, label="python")
    plt.plot(idl_x, idl_y, label="idl", ls=":")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(plot_output + title + '.png', dpi=dpi)
    plt.clf()


python_file = 'Results/output.npz'
idl_H_file = 'Results/IDL/test_kn1d.KN1D_H'
idl_H2_file = 'Results/IDL/test_kn1d.KN1D_H2'

#Loading files
print("loading python data:", python_file)
python_results = np.load(python_file)

print("loading H sav file:", idl_H_file)
idl_results = readsav(idl_H_file)
print("loading H2 sav file:", idl_H2_file)
idl_results.update(readsav(idl_H2_file))


#Copying IDL output to file
print("Printing IDL output data to idl_output.txt")

standard_out = sys.stdout
output = open('Results/idl_output.txt', 'w')
sys.stdout = output

for key, value in idl_results.items():
    print(key)
    print(value)
    print()
output.close()
sys.stdout = standard_out


#Creating output file for error values
output_pref = "Results/Output/"
output_file = input("Enter output file name: " + output_pref)
output_file = output_pref + output_file
if(output_file == output_pref):
    output_file += "output.txt"

output = open(output_file, 'w')
sys.stdout = output

for key in python_results:
    print()
    pred = np.asarray(python_results[key])
    act = np.asarray(idl_results[key])
    # n = len(act)
    if pred.size > 1:
        print(key, "Relative L2 Error: ", np.linalg.norm(pred-act, ord=2)/np.linalg.norm(pred))
    else:
        print(key, "Relative Error: ", perror(pred, act))
    # print(key, "max percentage error:", max_perror(python_results[key], idl_results[key]))
    # print(key, "mean percentage error:", mean_perror(python_results[key], idl_results[key]))

output.close()
sys.stdout = standard_out


print("\n")
print("Generating Plots")

dpi = 300

#Create Results Plots
generate_compare_plot("nH2", python_results["xH2"], python_results["nH2"], idl_results["xH2"], idl_results["nH2"], "x (m)", "Density (m^3)")

generate_compare_plot("nH", python_results["xH"], python_results["nH"], idl_results["xH"], idl_results["nH"], "x (m)", "Density (m^3)")

generate_compare_plot("Net H Source", python_results["xH"], python_results["NetHSource"], idl_results["xH"], idl_results["NetHSource"], "x (m)", "Net H Source (m^-3)")

generate_compare_plot("Relative H Source", python_results["xH"], (python_results["NetHSource"]/ python_results["nH"]),
                       idl_results["xH"], (idl_results["NetHSource"]/python_results["nH"]), "x (m)", "Relative H Source (m^-3)")