import torch
import numpy as np
import json
import time
from datetime import timedelta

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.kinetic_h import KineticH
from kn1ddiff.test.utils import *



dir = "kn1ddiff/test/beta_cx/"
image_dir = dir+"Images/"
data_file = "beta_cx_in_out.json"
epsilon = 10e-10
dtype = torch.float64

USE_CPU = False

OPTIMIZE_FH = True

# Learning Rate Parameters
INITIAL_LR = 1e-3
LR_CYCLE = 250
MIN_LR = 1e-6

# Iteration Parameters
NUM_ITERS = 1000
CLIP_NORM = 1e-1

# Gif parameters
GENERATE_GIF = True
GIF_FPS = 10
GIF_FREQ = 20


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and not USE_CPU else "cpu")
    print("device: ", device)
    # if use_cuda:
    #     torch.cuda.manual_seed(72)

    # --- Load Inputs and Outputs ---

    with open(dir+data_file, 'r') as f:
        data = json.load(f)

    # Fixed

    # Gradient
    true_fH = torch.tensor(data["fH"], dtype=dtype, device=device)
    # print("fH Range: ", torch.max(true_fH), torch.min(true_fH))

    # Desired Outputs
    
    true_beta_cx = torch.asarray(data["Beta_CX"], dtype=dtype, device=device)


    # --- Set up Kinetic_H ---

    with open(dir+"h_mesh_in.json", 'r') as f:
        mesh_input = json.load(f)
    with open(dir+"kinetic_h_in.json", 'r') as f:
        kh_in = json.load(f)

    for key, value in mesh_input.items():
        mesh_input[key] = np.asarray(value)
    for key, value in kh_in.items():
        kh_in[key] = torch.tensor(value, dtype=dtype, device=device)
    
    mesh = KineticMesh('h', mesh_input["mu"], mesh_input["x"], mesh_input["Ti"], mesh_input["Te"], mesh_input["n"], mesh_input["PipeDia"], E0=mesh_input["E0"], fctr=mesh_input["fctr"], device=device, dtype=dtype)
    
    kinetic_h = KineticH(mesh, kh_in["mu"], kh_in["vxi"], kh_in["fHBC"], kh_in["GammaxHBC"], 
                        ni_correct=True, truncate=1.0e-3, max_gen=100, 
                        compute_errors=True, debrief=True, debug=False, 
                        device=device, dtype=dtype)

    # kinetic_h internal Data
    kinetic_h.Internal.fi_hat = torch.tensor(data['fi_hat'], dtype=dtype, device=device)
    kinetic_h.Internal.Alpha_CX = torch.tensor(data['Alpha_CX'], dtype=dtype, device=device)
    kinetic_h.Internal.ni = torch.tensor(data['ni'], dtype=dtype, device=device)
    kinetic_h.Internal.SIG_CX = torch.tensor(data['SIG_CX'], dtype=dtype, device=device)


    # --- Test Input Data ---

    # beta_cx = kinetic_h._compute_beta_cx(true_fH)
    # print(torch.allclose(beta_cx, true_beta_cx))
    # print(rel_L2_torch(beta_cx, true_beta_cx))
    # input()


    # --- Test Optimization ---
    # initial_fH = torch.nn.Parameter(torch.randn_like(true_fH, requires_grad=True, dtype=dtype, device=device))
    initial_sign = torch.sign(true_fH.detach())
    initial_fH = torch.nn.Parameter(torch.log(torch.abs(0.9*torch.clone(true_fH.detach()))))
    # initial_fH = torch.nn.Parameter(torch.zeros_like(true_fH, requires_grad=True))

    parameters = []
    if OPTIMIZE_FH:
        parameters.append(initial_fH)

    optimizer = torch.optim.Adam(parameters, lr=INITIAL_LR, betas=(0.9,  0.99))


    # --- Scheduler Options --- 

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     factor=0.1,
    #     patience=50,
    #     min_lr=1e-5
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_ITERS)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=LR_CYCLE,
        # T_mult=1,
        # eta_min=MIN_LR,
    )


    # --- Loss Function Options --- 

    # loss_fun = lambda pred, true : torch.nn.functional.mse_loss(pred, true)
    # loss_fun = lambda p, t: torch.mean((p - t)**2 / (t**2 + 1e-12))
    # loss_fun = lambda pred, true : rel_L2_torch(pred, true)
    # loss_fun = lambda pred, true : torch.mean(
    #         (torch.log(torch.abs(pred) + 1e-12)
    #     - torch.log(torch.abs(true) + 1e-12))**2
    #     )
    # loss_fun = lambda pred, true : ((torch.log(pred + epsilon) - torch.log(true + epsilon))**2).mean()

    def symmetric_log(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    loss_fun = lambda pred, true : ((symmetric_log(pred) - symmetric_log(true))**2).mean()










    # --- Optimization ---

    # Init Gif Generator
    fh_gifgen = GIF_Generator(NUM_ITERS, image_dir+"fH_Images/", "fH", true_fH[0,10,:], fps=GIF_FPS, frequency=GIF_FREQ)

    # Capture Best Epoch
    loss_list = []
    lr_list = []
    best_loss = np.inf
    best_epoch = 0

    optim_start = time.time()

    for epoch in range(NUM_ITERS):

        epoch_start = time.time()

        if OPTIMIZE_FH:
            # fH = 1e19 * torch.nn.functional.softplus(initial_fH)
            # fH = 1e19 * torch.nn.functional.tanh(initial_fH)
            # fH = 1e19 * torch.sigmoid(initial_fH)
            fH = initial_sign*torch.exp(initial_fH)
        else:
            fH = true_fH

    
        beta_cx = kinetic_h._compute_beta_cx(fH)

        # Compute Loss
        loss = loss_fun(beta_cx, true_beta_cx)

        # Backprop
        optimizer.zero_grad()
        loss.backward()

        # Clip Gradient
        torch.nn.utils.clip_grad_norm_([initial_fH], max_norm=CLIP_NORM)

        #Optimize
        optimizer.step()
        # scheduler.step(loss)
        scheduler.step()

        # Save Best Epoch
        loss_list.append(loss.item())
        lr_list.append(scheduler.get_last_lr())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_inputs = [fH.detach()]
            best_pred = [beta_cx.detach()]
            best_epoch = epoch

        epoch_runtime = time.time() - epoch_start

        print(
            f"epoch: {epoch:<5} | "
            f"runtime: {epoch_runtime:<8.2} | "
            f"loss: {loss.item():<10.6e} | "
            f"learning rate: {scheduler.get_last_lr()[0]:.2e}"
        )

        # print("FH_2", fH[0,10,:])
        if GENERATE_GIF:
            fh_gifgen.update(fH[0,10,:], epoch)

    optimization_runtime = time.time() - optim_start
    print(f"Total Optimization Time: {timedelta(seconds=round(optimization_runtime))}")








    # --- Analysis ---

    opt_fH = best_inputs[0]
    opt_beta_cx = best_pred[0]

    fH_loss = loss_fun(opt_fH, true_fH).item()
    
    beta_cx_loss = loss_fun(opt_beta_cx, true_beta_cx).item()

    # --- Analyze ---
    print("Best Epoch: ", best_epoch)

    if OPTIMIZE_FH:
        print("fH Loss: ", fH_loss)
        print("fH Relative L2: ", rel_L2_torch(opt_fH, true_fH))
        print()


    print("Beta_CX Loss: ", beta_cx_loss)
    print("Beta_CX Relative L2: ", rel_L2_torch(opt_beta_cx, true_beta_cx))



    print("Generating Images and Gifs")


    x = range(opt_fH[0,10,:].numel())
    for i in range(len(opt_fH[0,:,0])):
        generate_compare_plot(image_dir+"fH_Images/", "fH_slice"+str(i), x, opt_fH[0,i,:], x, true_fH[0,i,:])

    generate_loss_plot(image_dir, "Loss", loss_list, xlabel="Epoch", ylabel="Symmetrical Loss")
    generate_lr_plot(image_dir, "LR", lr_list, xlabel="Epoch", ylabel="Learning Rate")

    if GENERATE_GIF:
        if OPTIMIZE_FH:
            fh_gifgen.generate_gif()