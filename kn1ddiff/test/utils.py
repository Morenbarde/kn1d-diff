import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def generate_compare_plot(dir, title, x, y, true_x, true_y, xlabel="", ylabel="", x_range = None, y_range = None):
    check_and_generate_dir(dir)
    plt.plot(x, y, color = 'blue', marker='x', markersize=3, markeredgecolor='cyan', label="Optimized")
    plt.plot(true_x, true_y, color = 'orange', marker='x', markersize=3, markeredgecolor='red', label="True", ls=":")
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




def rel_L2_np(pred, act, eps=1e-12):
    num = np.linalg.norm(pred - act)
    den = np.linalg.norm(pred)

    return num / (den + eps)

def rel_L2_torch(pred, act, eps=1e-12):
    num = torch.linalg.norm(pred - act)
    den = torch.linalg.norm(pred)

    return num / (den + eps)



def check_and_generate_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class GIF_Generator():

    def __init__(self, num_epochs, target_dir, name, true_val: torch.Tensor, frequency=1, fps=24):

        self.frequency = frequency
        self.size = num_epochs // frequency
        self.current_epoch = 0

        if true_val.ndim != 1:
            print("WARNING: This class only supports 1D data, using first dimension of the array")
            true_val = true_val[:1]
        self.true_val = true_val.detach().cpu().numpy()

        self.data_size = self.true_val.size
        self.data = np.empty((self.size, self.data_size))

        self.image_location = target_dir
        self.name = name
        self.fps = fps

        self.image_paths = []

    def update(self, new_data: torch.Tensor, epoch):
        if new_data.ndim != 1:
            print("WARNING: This class only supports 1D data, using first dimension of the array")
            new_data = new_data[:1]
        self.data[epoch//self.frequency] = new_data.detach().cpu().numpy()

    def generate_gif(self):
        print("Generating "+self.name+" GIF")
        self._generate_images()
        self._animate_images()
        self._remove_images()


    def _generate_images(self):
        x = range(self.data_size)
        ymin = min(np.min(self.data[-1]), np.min(self.true_val))
        ymax = max(np.max(self.data[-1]), np.max(self.true_val))
        data_range = ymax-ymin
        ymin = ymin-0.05*data_range
        ymax = ymax+0.05*data_range
        for i in range(self.size):
            num_name = self.name+"_"+str(i)
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