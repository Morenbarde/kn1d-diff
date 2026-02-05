import numpy as np
from scipy.io import readsav
import matplotlib.pyplot as plt
import json
import os

data_file = './sav_files/kn1d_test_inputs.sav'
# data_file = './sav_files/1090904018_950to1050.sav'
# data_file = './sav_files/1090904029_950to1050_towall.sav'
print("Loading file: "  + data_file)
sav_data = readsav(data_file)

def rel_L2(pred, act):
    pred = np.asarray(pred)
    act  = np.asarray(act)

    if pred.size == 1:
        num = abs((pred-act).item())
        den = abs(pred.item())
    else:
        num = np.linalg.norm(pred-act, ord=2)
        den = np.linalg.norm(pred)
    if den == 0:
        return num #Absolute Error
    return num/den

def generate_compare_plot(dir, title, x, y, idl_x, idl_y, xlabel, ylabel):
    plt.plot(x, y, color = 'blue', marker='x', markersize=3, markeredgecolor='cyan', label="KN1DPy")
    plt.plot(idl_x, idl_y, color = 'orange', marker='x', markersize=3, markeredgecolor='red', label="Francesco's IDL", ls=":")
    plt.axvline(x=sav_data['x_lim'], color='black', linestyle='--', linewidth=1, label="x_lim")
    plt.axvline(x=sav_data['x_sep'], color='black', linestyle=':', linewidth=1, label="x_sep")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(dir + title + '.png', dpi=300)
    plt.clf()

def generate_l2_bar_plot(dir, errors):

    outputs = list(errors.keys())
    scores = []
    for key in outputs:
        scores.append(errors[key])

    font_size = 20
    label_fontsize = 15
        
    # Bar Graphs
    fig, ax = plt.subplots(figsize=(16,9))
    bars = ax.bar(outputs, scores, width=0.8)
    ax.set_yscale('log')
    plt.title("KN1DPy vs KN1D Relative L2 Errors", fontsize=font_size)
    plt.xticks(fontsize=label_fontsize, rotation='vertical')
    plt.subplots_adjust(bottom=0.2)
    plt.yticks(fontsize=label_fontsize)
    # plt.set_yscale('log')
    plt.ylabel("Relative L2 Error", fontsize=font_size)
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor('black')
    plt.savefig(dir+"l2_scores.png")
    plt.clf()


def generate_all_plots(python_results, idl_results, l2_errors, output_dir):
    plot_dir = output_dir+'AutoPlots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    h_len = np.asarray(python_results["xH"]).size
    h2_len = np.asarray(python_results["xH2"]).size

    for key in python_results:
        if key == "xH" or key == "xH2" or key == "runtime" or key == "GammaHLim":
            continue
        py_y = np.asarray(python_results[key])
        idl_y = np.asarray(idl_results[key])

        if py_y.size == h_len:
            py_x = np.asarray(python_results["xH"])
            idl_x = np.asarray(idl_results["xH"])
        elif py_y.size == h2_len:
            py_x = np.asarray(python_results["xH2"])
            idl_x = np.asarray(idl_results["xH2"])

        generate_compare_plot(plot_dir, key, py_x, py_y, idl_x, idl_y, "x (m)", key)

    generate_l2_bar_plot(plot_dir, l2_errors)



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


# Gather Run Data
run_name = input("Run Name: ")

python_runtime = 1 #float(input("Python Runtime: "))
idl_runtime = 1 #float(input("IDL Runtime: "))


python_dict = {}
idl_dict = {}
L2_dict = {}

# Store Results
python_dict["runtime"] = python_runtime
idl_dict["runtime"] = idl_runtime

for key in python_results:
    print("Storing Key: ", key)
    pred = np.asarray(python_results[key])
    act = np.asarray(idl_results[key])

    # Store as list for json conversion
    python_dict[key] = pred.tolist()
    idl_dict[key] = act.tolist()

    L2_dict[key] = rel_L2(pred, act)

#Output Results
output_dir = "Results/Runs/"+run_name+"/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_dir+'python_output.json', "w") as py_file:
    json.dump(python_dict, py_file, indent=4)

with open(output_dir+'idl_output.json', "w") as idl_file:
    json.dump(idl_dict, idl_file, indent=4)

with open(output_dir+'L2_output.json', "w") as l2_file:
    json.dump(L2_dict, l2_file, indent=4)

generate_all_plots(python_dict, idl_dict, L2_dict, output_dir)