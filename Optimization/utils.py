import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from PIL import Image


def init_optimization_tensor(tensor, factor, offset_factor):
    tensor = torch.clone(tensor.detach())
    return (factor*tensor) + (offset_factor*tensor)

def rel_L2_np(pred, act, eps=1e-12):
    num = np.linalg.norm(pred - act)
    den = np.linalg.norm(pred)

    return num / (den + eps)

def rel_L2_torch(pred, act, eps=1e-12):
    num = torch.linalg.norm(pred - act)
    den = torch.linalg.norm(pred)

    return (num / (den + eps)).item()

def rel_L2_loss(pred, act, eps=1e-12):
    num = torch.linalg.norm(pred - act)
    den = torch.linalg.norm(pred)

    return (num / (den + eps))

def make_json_compatible(data: dict):
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = torch_to_numpy(value).tolist()
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()
        if isinstance(value, np.float32) or isinstance(value, np.float64):
            data[key] = float(value)
    return data

def sav_to_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


# Analysis

def check_close(var_name, pred, true):
    print(var_name+" close: ", torch.allclose(pred, true))
    print(var_name+" L2: ", rel_L2_torch(pred, true))

def analyze_difference(name, loss_fun, pred, true):
    loss = loss_fun(pred, true).item()
    l2 = rel_L2_torch(pred, true)
    print(name+" Loss: ", loss)
    print(name+" Relative L2: ", l2)
    return loss, l2

def print_torch_range(tensor: torch.Tensor, var_name = "Var"):
    print(var_name+" Range: ", torch.min(tensor).item(), torch.max(tensor).item())
    print(var_name+" Mean/Med: ", torch.mean(tensor).item(), torch.median(tensor).item())


# Torch converter
def numpy_to_torch(np_arr, device, dtype):
    return torch.from_numpy(np_arr).to(dtype=dtype, device=device)

def torch_to_numpy(torch_tensor):
    if type(torch_tensor) == torch.Tensor:
        torch_tensor = torch_tensor.cpu().detach().numpy()
    return torch_tensor


# Directories

def check_and_generate_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def rename_dir(old_dir, new_dir):
    os.rename(old_dir, new_dir)


# Logging

def setup_log(run_dir):
    check_and_generate_dir(run_dir)

    log_path = os.path.join(run_dir, "optim.log")
    log_file = open(log_path, "a", buffering=1)

    sys.stdout = log_file
    sys.stderr = log_file

    

# Plotting

def generate_compare_plot(dir, title, x, y, true_x, true_y, init_x = None, init_y = None, xlabel="", ylabel="", x_range = None, y_range = None):
    check_and_generate_dir(dir)
    
    # Adjust types if necessary
    torch_to_numpy(x)
    torch_to_numpy(y)
    torch_to_numpy(true_x)
    torch_to_numpy(true_y)
    torch_to_numpy(init_x)
    torch_to_numpy(init_y)

    plt.plot(x, y, color = 'blue', marker='x', markersize=2, markeredgecolor='cyan', label="Optimized")
    plt.plot(true_x, true_y, color = 'orange', marker='x', markersize=2, markeredgecolor='red', label="True", ls=":")
    if(init_x is not None and init_y is not None):
        plt.plot(init_x, init_y, color = 'pink', marker='x', markersize=2, markeredgecolor='purple', label="Initial", ls=":")
    # plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    ax = plt.gca()
    if(x_range):
        ax.set_xlim(x_range)
    if(y_range):
        ax.set_ylim(y_range)
    plt.savefig(dir + title + '.png', dpi=300)
    plt.clf()

def generate_grad_plot(dir, title, x, y, xlabel="", ylabel="", x_range = None, y_range = None):
    check_and_generate_dir(dir)
    
    # Adjust types if necessary
    torch_to_numpy(x)
    torch_to_numpy(y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    ax = plt.gca()
    if(x_range):
        ax.set_xlim(x_range)
    if(y_range):
        ax.set_ylim(y_range)
    plt.savefig(dir + title + '.png', dpi=300)
    plt.clf()

def generate_loss_plot(dir, title, loss, xlabel="", ylabel=""):
    check_and_generate_dir(dir)
    plt.plot(range(len(loss)), loss, color = 'purple')
    plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(dir + title + '.png', dpi=300)
    plt.clf()

def generate_lr_plot(dir, title, lr, xlabel="", ylabel=""):
    check_and_generate_dir(dir)
    plt.plot(range(len(lr)), lr, color = 'teal')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(dir + title + '.png', dpi=300)
    plt.clf()


class GIF_Generator():

    def __init__(self, num_epochs, target_dir, name, true_val: torch.Tensor = None, frequency=1, fps=24):

        self.frequency = frequency
        self.size = num_epochs // frequency
        self.current_epoch = 0

        if true_val is not None:
            if true_val.ndim != 1:
                print("WARNING: This class only supports 1D data, using first dimension of the array")
                true_val = true_val[:1]
            self.true_val = true_val.detach().cpu().numpy()
            self.data_size = self.true_val.size
        else:
            self.true_val = None
            self.data_size = None

        # self.data_size = self.true_val.size
        self.data = []

        self.image_location = target_dir
        self.name = name
        self.fps = fps

        self.image_paths = []

    def update(self, new_data: torch.Tensor, epoch):
        if new_data.ndim != 1:
            print("WARNING: This class only supports 1D data, using first dimension of the array")
            new_data = new_data[:1]
        self.data.append(new_data.detach().cpu().numpy())

    def generate_gif(self):
        print("Generating "+self.name+" GIF")
        self.data = np.array(self.data)
        self._generate_images()
        self._animate_images()
        self._remove_images()


    def _generate_images(self):
        x = range(len(self.data[0]))
        # ymin = min(np.min(self.data[-1]), np.min(self.true_val))
        # ymax = max(np.max(self.data[-1]), np.max(self.true_val))
        if self.true_val is None:
            ymin = -20
            ymax = 20
        else:
            ymin = np.min(self.true_val)
            ymax = np.max(self.true_val)
            data_range = ymax-ymin
            ymin = ymin-0.05*data_range
            ymax = ymax+0.05*data_range
        for i in range(self.size):
            num_name = self.name+"_epoch_"+str(np.floor(i*self.frequency))
            if self.true_val is None:
                generate_grad_plot(self.image_location, num_name, x, self.data[i])
            else:
                generate_compare_plot(self.image_location, num_name, x, self.data[i], x, self.true_val, y_range=[ymin, ymax])
            self.image_paths.append(self.image_location+num_name+".png")
        print("Images Generated")

    def _animate_images(self):
        print("Generating GIF")
        # Open images
        images = [Image.open(path) for path in self.image_paths]

        # Save as animated GIF
        images[0].save(
            self.image_location+self.name+".gif",
            save_all=True,
            append_images=images[1:],
            duration=1000/self.fps,  # Duration per frame in milliseconds
            loop=0         # 0 = loop forever
        )

    def _remove_images(self):
        print("Removing Images")

        for file_name in self.image_paths:
            try:
                os.remove(file_name)
                print(f"Deleted: {file_name}")
            except FileNotFoundError:
                print(f"File not found: {file_name}")