import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import timedelta

from kn1ddiff.create_shifted_maxwellian import *
from KN1DPy.create_shifted_maxwellian import create_shifted_maxwellian as csm
from kn1ddiff.test.utils import *



dir = "kn1ddiff/test/shift_maxwell/"
data_file = "in_out2.npz"
generate_gif = True
num_iters = 200

dtype = torch.float64


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)
    # if use_cuda:
    #     torch.cuda.manual_seed(72)

    # torch.set_default_dtype(torch.float64)
    
    data = np.load("kn1ddiff/test/shift_maxwell/"+data_file)
    print(data["mol"])
    print(data["mu"])


    # --- Load Inputs and Outputs ---

    # Fixed
    vx = numpy_to_torch(data["vx"], device, dtype)
    vr = numpy_to_torch(data["vr"], device, dtype)
    mu = numpy_to_torch(data["mu"], device, dtype)
    mol = numpy_to_torch(data["mol"], device, dtype)
    Tnorm = numpy_to_torch(data["Tnorm"], device, dtype)

    # Gradient
    Tmaxwell = numpy_to_torch(data["Tmaxwell"], device, dtype)
    vx_shift = numpy_to_torch(data["vx_shift"], device, dtype)

    # Desired Outputs
    maxwell_old = numpy_to_torch(data["maxwell"], device, dtype)
    # with torch.no_grad():
    #     maxwell_old2 = create_shifted_maxwellian(vr, vx, Tmaxwell, vx_shift, mu, mol, Tnorm)
    #     maxwell_old3 = csm(data["vr"], data["vx"], data["Tmaxwell"], data["vx_shift"], data["mu"], data["mol"], data["Tnorm"])

    # print(rel_L2_torch(maxwell_old2, maxwell_old3).item())
    # # for i in range(vx_shift.numel()):
    # #     print(rel_L2_torch(maxwell_old2[:,:,i], maxwell_old3[:,:,i]).item())
    # input()


    # --- Test Optimization ---
    initial_tmaxwell = torch.nn.Parameter(torch.randn_like(Tmaxwell, device=device, dtype=dtype))
    initial_vx_shift = torch.nn.Parameter(torch.randn_like(vx_shift, device=device, dtype=dtype))

    optimizer = torch.optim.Adam([initial_tmaxwell, initial_vx_shift], lr=2e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=30,
        min_lr=1e-6
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)

    # Init Gif Generator
    tmax_gifgen = GIF_Generator(num_iters, dir+"Tmax_Images/", "tmax", Tmaxwell, fps=10, frequency=10)
    vx_shift_gifgen = GIF_Generator(num_iters, dir+"VxShift_Images/", "vx_shift", vx_shift, fps=10, frequency=10)

    # Capture Best Epoch
    loss_list = []
    best_loss = np.inf
    best_epoch = 0

    optim_start = time.time()

    for epoch in range(num_iters):

        epoch_start = time.time()

        tmax = 600 * torch.sigmoid(initial_tmaxwell)
        shift = 5000 * torch.sigmoid(initial_vx_shift)

        maxwell = create_shifted_maxwellian(vr, vx, tmax, shift, mu, mol, Tnorm)

        # Compute Loss
        loss = torch.nn.functional.mse_loss(maxwell, maxwell_old)
        # loss = rel_L2_torch(maxwell, maxwell_old)


        # Backprop
        optimizer.zero_grad()
        loss.backward()

        # Clip Gradient
        torch.nn.utils.clip_grad_norm_([initial_tmaxwell, initial_vx_shift], max_norm=1.0)

        #Optimize
        optimizer.step()
        scheduler.step(loss)

        # Save Best Epoch
        loss_list.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_inputs = [tmax.detach(), shift.detach()]
            best_pred = maxwell.detach()
            best_epoch = epoch

        epoch_runtime = time.time() - epoch_start

        print(
            f"epoch: {epoch:<5} | "
            f"runtime: {epoch_runtime:<8.2} | "
            f"loss: {loss.item():<10.6e} | "
            f"learning rate: {optimizer.param_groups[0]['lr']:.2e} "
        )

        if generate_gif:
            tmax_gifgen.update(tmax, epoch)
            vx_shift_gifgen.update(shift, epoch)

    
    optimization_runtime = time.time() - optim_start
    print(f"Total Optimization Time: {timedelta(seconds=round(optimization_runtime))}")


    # --- Convert to numpy for analysis ---

    tmax, shift = best_inputs[0], best_inputs[1]
    maxwell = best_pred

    tmax_loss = torch.nn.functional.mse_loss(tmax, Tmaxwell).item()
    vxshift_loss = torch.nn.functional.mse_loss(shift, vx_shift).item()
    max_loss = torch.nn.functional.mse_loss(maxwell, maxwell_old).item()

    true_tmax = torch_to_numpy(Tmaxwell)
    opt_tmax = torch_to_numpy(tmax)

    true_vxshift = torch_to_numpy(vx_shift)
    opt_vxshift = torch_to_numpy(shift)

    true_maxwell = torch_to_numpy(maxwell_old)
    opt_maxwell = torch_to_numpy(maxwell)


    # --- Analyze ---
    print("Best Epoch: ", best_epoch)

    print("True Tmaxwell Range")
    print(np.min(true_tmax), np.max(true_tmax))
    print("Calculated Tmaxwell Range")
    print(np.min(opt_tmax), np.max(opt_tmax))
    print()
    print("True vx_shift range")
    print(np.min(true_vxshift), np.max(true_vxshift))
    print("Calculated vx_shift range")
    print(np.min(opt_vxshift), np.max(opt_vxshift))
    # input()

    print("True Tmaxwell")
    print(true_tmax)
    print("Calculated Tmaxwell")
    print(opt_tmax)
    print("Tmaxwell Loss: ", tmax_loss)
    print("Tmaxwell Relative L2: ", rel_L2_np(opt_tmax, true_tmax))
    print()
    print("True vx_shift")
    print(true_vxshift)
    print("Calculated vx_shift")
    print(opt_vxshift)
    print("vx_shift Loss: ", vxshift_loss)
    print("vx_shift Relative L2: ", rel_L2_np(opt_vxshift, true_vxshift))

    # input()

    # print(maxwell_old[0])
    # print(maxwell[0])

    print("True Maxwell Sum:       ", np.sum(true_maxwell))
    print("Calculated Maxwell Sum: ", np.sum(opt_maxwell))
    print("Maxwell Loss: ", max_loss)
    print("Maxwell Relative L2: ", rel_L2_np(opt_maxwell, true_maxwell))
    # input()



    print("Generating Images and Gifs")

    x = range(opt_tmax.size)
    generate_compare_plot(dir, "Tmaxwell", x, opt_tmax, x, true_tmax)
    x = range(opt_vxshift.size)
    generate_compare_plot(dir, "vx_shift", x, opt_vxshift, x, true_vxshift)

    generate_loss_plot(dir, "Loss", loss_list, xlabel="Epoch", ylabel="MSE Loss")

    tmax_gifgen.generate_gif()
    vx_shift_gifgen.generate_gif()



