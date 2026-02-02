import torch
import numpy as np
import matplotlib.pyplot as plt

from kn1ddiff.create_shifted_maxwellian import *


def generate_compare_plot(dir, title, x, y, true_x, true_y, xlabel="", ylabel=""):
    plt.plot(x, y, color = 'blue', marker='x', markersize=3, markeredgecolor='cyan', label="Optimized")
    plt.plot(true_x, true_y, color = 'orange', marker='x', markersize=3, markeredgecolor='red', label="True", ls=":")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(dir + title + '.png', dpi=300)
    plt.clf()



dir = "kn1ddiff/test/shift_maxwell/"
data_file = "in_out2.npz"
num_iters = 500


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)
    # if use_cuda:
    #     torch.cuda.manual_seed(72)

    torch.set_default_dtype(torch.float64)
    
    data = np.load("kn1ddiff/test/shift_maxwell/"+data_file)
    print(data["mol"])
    print(data["mu"])


    # --- Load Inputs and Outputs ---

    # Fixed
    vx = torch.from_numpy(data["vx"])
    vr = torch.from_numpy(data["vr"])
    mu = torch.from_numpy(data["mu"])
    mol = torch.from_numpy(data["mol"])
    Tnorm = torch.from_numpy(data["Tnorm"])

    # Gradient
    Tmaxwell = torch.from_numpy(data["Tmaxwell"])
    vx_shift = torch.from_numpy(data["vx_shift"])

    # Desired Outputs
    maxwell_old = torch.from_numpy(data["maxwell"])
    with torch.no_grad():
        maxwell_old2 = create_shifted_maxwellian(vr, vx, Tmaxwell, vx_shift, mu, mol, Tnorm)

    # print(torch.nn.functional.mse_loss(maxwell_old, maxwell_old2).item())
    # input()


    # --- Test Optimization ---
    initial_tmaxwell = torch.nn.Parameter(torch.randn_like(Tmaxwell))
    initial_vx_shift = torch.nn.Parameter(torch.randn_like(vx_shift))

    optimizer = torch.optim.Adam([initial_tmaxwell, initial_vx_shift], lr=1.5e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=30,
        min_lr=1e-6
    )

    for epoch in range(num_iters):
        tmax = 600 * torch.sigmoid(initial_tmaxwell)
        shift = 5000 * torch.sigmoid(initial_vx_shift)
        # tmax = 1 * torch.nn.functional.softplus(initial_tmaxwell)
        # shift = 5e3 * torch.nn.functional.softplus(initial_vx_shift)

        maxwell = create_shifted_maxwellian(vr, vx, tmax, shift, mu, mol, Tnorm)

        # Compute Loss
        loss = torch.nn.functional.mse_loss(maxwell, maxwell_old)


        # Backprop
        optimizer.zero_grad()
        loss.backward()

        # Clip Gradient
        torch.nn.utils.clip_grad_norm_([initial_tmaxwell, initial_vx_shift], max_norm=1.0)

        #Optimize
        optimizer.step()
        scheduler.step(loss)

        print(
            f"epoch: {epoch:<5} | "
            f"loss: {loss.item():<10.6e} | "
            f"learning rate: {optimizer.param_groups[0]['lr']:.2e}"
        )


    # --- Convert to numpy for analysis ---

    tmax_loss = torch.nn.functional.mse_loss(tmax, Tmaxwell).item()
    vxshift_loss = torch.nn.functional.mse_loss(shift, vx_shift).item()
    max_loss = torch.nn.functional.mse_loss(maxwell, maxwell_old).item()

    true_tmax = Tmaxwell.cpu().detach().numpy()
    opt_tmax = tmax.cpu().detach().numpy()

    true_vxshift = vx_shift.cpu().detach().numpy()
    opt_vxshift = shift.cpu().detach().numpy()

    true_maxwell = maxwell_old.cpu().detach().numpy()
    opt_maxwell = maxwell.cpu().detach().numpy()


    # --- Analyze ---

    print("True Tmaxwell Range")
    print(np.min(true_tmax), np.max(true_tmax))
    print("Calculated Tmaxwell Range")
    print(np.min(opt_tmax), np.max(opt_tmax))
    print()
    print("True vx_shift range")
    print(np.min(true_vxshift), np.max(true_vxshift))
    print("Calculated vx_shift range")
    print(np.min(opt_vxshift), np.max(opt_vxshift))
    input()

    print("True Tmaxwell")
    print(true_tmax)
    print("Calculated Tmaxwell")
    print(opt_tmax)
    print("Tmaxwell Loss: ", tmax_loss)
    print()
    print("True vx_shift")
    print(true_vxshift)
    print("Calculated vx_shift")
    print(opt_vxshift)
    print("vx_shift Loss: ", vxshift_loss)

    x = range(opt_tmax.size)
    generate_compare_plot(dir, "Tmaxwell", x, opt_tmax, x, true_tmax)
    x = range(opt_vxshift.size)
    generate_compare_plot(dir, "vx_shift", x, opt_vxshift, x, true_vxshift)

    input()

    # print(maxwell_old[0])
    # print(maxwell[0])

    print("True Maxwell Sum:       ", np.sum(true_maxwell))
    print("Calculated Maxwell Sum: ", np.sum(opt_maxwell))
    print("Maxwell Loss: ", max_loss)



