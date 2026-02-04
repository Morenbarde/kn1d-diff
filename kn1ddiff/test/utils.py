import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_compare_plot(dir, title, x, y, true_x, true_y, xlabel="", ylabel=""):
    plt.plot(x, y, color = 'blue', marker='x', markersize=3, markeredgecolor='cyan', label="Optimized")
    plt.plot(true_x, true_y, color = 'orange', marker='x', markersize=3, markeredgecolor='red', label="True", ls=":")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(dir + title + '.png', dpi=300)
    plt.clf()

def generate_loss_plot(dir, title, loss, xlabel="", ylabel=""):
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