import torch
import numpy as np
import json
import time
from datetime import timedelta
import math

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.kinetic_h import *
from kn1ddiff.test.utils import *



dir = "kn1ddiff/test/omega_vals/"
image_dir = dir+"Images/"
data_file = "omega_in_out.json"

# Torch
dtype = torch.float64
USE_CPU = False

# Constants
EPSILON = 10e-10

# Optimization Choices
OPTIMIZE_FH = True
OPTIMIZE_NH = True

# Iteration Parameters
NUM_ITERS = 500
CLIP_NORM = 1e-0

# Learning Rate Parameters
INITIAL_LR = 5e-4
LR_CYCLE_COUNT = 1
LR_CYCLE = math.ceil(NUM_ITERS // LR_CYCLE_COUNT)
MIN_LR = 1e-5

# Gif parameters
GENERATE_GIF = True
GIF_FPS = 10
GIF_FREQ = 5


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and not USE_CPU else "cpu")
    print("device: ", device)
    # if use_cuda:
    #     torch.cuda.manual_seed(72)

    # torch.autograd.set_detect_anomaly(True)


    # --- Load Inputs and Outputs ---
    with open(dir+data_file, 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            data[key] = torch.tensor(value, dtype=dtype, device=device)

    # Fixed

    # Gradient
    truein_fH = data["fH"]
    print("fH Range: ", torch.max(truein_fH), torch.min(truein_fH))
    truein_nH = data["nH"]
    print("nH Range: ", torch.max(truein_nH), torch.min(truein_nH))

    # Desired Outputs
    
    trueout_OH_H = data["OH_H"]
    trueout_OH_P = data["OH_P"]
    trueout_OH_H2 = data["OH_H2"]


    # --- Set up Kinetic_H ---
    
    # --- Set up Kinetic_H ---

    with open(dir+"h_mesh_in.json", 'r') as f:
        mesh_input = json.load(f)
        for key, value in mesh_input.items():
            mesh_input[key] = np.asarray(value)

    with open(dir+"kinetic_h_in.json", 'r') as f:
        kh_in = json.load(f)
        for key, value in kh_in.items():
            kh_in[key] = torch.tensor(value, dtype=dtype, device=device)
    
    mesh = KineticMesh('h', mesh_input["mu"], mesh_input["x"], mesh_input["Ti"], mesh_input["Te"], mesh_input["n"], mesh_input["PipeDia"], E0=mesh_input["E0"], fctr=mesh_input["fctr"], device=device, dtype=dtype)
    
    kinetic_h = KineticH(mesh, kh_in["mu"], kh_in["vxi"], kh_in["fHBC"], kh_in["GammaxHBC"], 
                        ni_correct=True, truncate=1.0e-3, max_gen=100, 
                        compute_errors=True, debrief=True, debug=False, 
                        device=device, dtype=dtype)

    # kinetic_h internal Data
    kinetic_h.H2_Moments.VxH2 = data['VxH2']
    kinetic_h.Internal.Alpha_H_H2 = data['Alpha_H_H2']
    kinetic_h.Internal.Alpha_H_P = data['Alpha_H_P']
    kinetic_h.Internal.MH_H_sum = data['MH_H_sum']


    # --- Test Input Data ---

    omega_vals = kinetic_h._compute_omega_values(truein_fH, truein_nH)

    # print("OH_H close: ", torch.allclose(trueout_OH_H, omega_vals.H_H))
    # print("OH_H L2: ", rel_L2_torch(trueout_OH_H, omega_vals.H_H))
    # print("OH_H range: ", torch.min(omega_vals.H_H), torch.max(omega_vals.H_H))

    # print("OH_P close: ", torch.allclose(trueout_OH_P, omega_vals.H_P))
    # print("OH_P L2: ", rel_L2_torch(trueout_OH_P, omega_vals.H_P))
    # print("OH_P range: ", torch.min(omega_vals.H_P), torch.max(omega_vals.H_P))

    # print("OH_H2 close: ", torch.allclose(trueout_OH_H2, omega_vals.H_H2))
    # print("OH_H2 L2: ", rel_L2_torch(trueout_OH_H2, omega_vals.H_H2))
    # print("OH_H2 range: ", torch.min(omega_vals.H_H2), torch.max(omega_vals.H_H2))
    # input()


    # --- Optimization Parameters ---

    initial_fH = 1.1*torch.clone(truein_fH.detach())
    fH_param = torch.nn.Parameter(torch.log(torch.abs(initial_fH)))
    initial_nH = 1.1*torch.clone(truein_nH.detach())
    nH_param = torch.nn.Parameter(torch.log(torch.abs(initial_nH)))

    parameters = []
    if OPTIMIZE_FH:
        parameters.append(fH_param)
    if OPTIMIZE_NH:
        parameters.extend([nH_param])

    optimizer = torch.optim.Adam(parameters, lr=INITIAL_LR, betas=(0.9,  0.99))


    # --- Scheduler Options --- 

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     factor=0.1,
    #     patience=50,
    #     min_lr=1e-5
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_ITERS)

    print("Learning Rate Cycle: ", LR_CYCLE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=LR_CYCLE,
        # T_mult=1,
        # eta_min=MIN_LR,
    )


    # --- Loss Function Options --- 

    # loss_fun = lambda pred, true : torch.nn.functional.mse_loss(pred, true)
    # loss_fun = lambda p, t: torch.mean((p - t)**2 / (t**2 + 1e-12))
    loss_fun = lambda pred, true : rel_L2_loss(pred, true)
    # loss_fun = lambda pred, true : torch.mean(
    #         (torch.log(torch.abs(pred) + 1e-12)
    #     - torch.log(torch.abs(true) + 1e-12))**2
    #     )
    # loss_fun = lambda pred, true : ((torch.log(pred + EPSILON) - torch.log(true + EPSILON))**2).mean()

    # def symmetric_log(x):
    #     return torch.sign(x) * torch.log1p(torch.abs(x))
    # loss_fun = lambda pred, true : ((symmetric_log(pred) - symmetric_log(true))**2).mean()










    # --- Optimization ---

    # Init Gif Generator
    if GENERATE_GIF:
        if OPTIMIZE_FH:
            fh_gifgen = GIF_Generator(NUM_ITERS, image_dir+"fH/", "fH", truein_fH[5,9,:], fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_NH:
            nh_gifgen = GIF_Generator(NUM_ITERS, image_dir+"nH/", "nH", truein_nH, fps=GIF_FPS, frequency=GIF_FREQ)


    # Capture Best Epoch
    loss_list = []
    lr_list = []
    best_loss = np.inf
    best_epoch = 0

    optim_start = time.time()

    for epoch in range(NUM_ITERS):

        epoch_start = time.time()

        # --- Bound Inputs ---

        if OPTIMIZE_FH:
            fH_in = torch.sign(initial_fH)*torch.exp(fH_param)
        else:
            fH_in = truein_fH

        if OPTIMIZE_NH:
            nH_in = torch.exp(nH_param)
        else:
            nH_in = truein_nH


        # --- Run Function ---
        omega_vals = kinetic_h._compute_omega_values(fH_in, nH_in)

        forward_done = time.time()
        print("Forward Time: ", forward_done - epoch_start)


        # --- Optimize ---

        # Compute Loss
        loss1 = loss_fun(omega_vals.H_H, trueout_OH_H)
        loss2 = loss_fun(omega_vals.H_P, trueout_OH_P)
        loss3 = loss_fun(omega_vals.H_H2, trueout_OH_H2)
        
        loss = loss1 + loss2 + loss3

        # Backprop
        optimizer.zero_grad()
        loss.backward()

        print(fH_param.grad.abs().mean())
        # print(nH_param.grad.abs().mean())

        # Clip Gradient
        # torch.nn.utils.clip_grad_norm_([fH_param], max_norm=CLIP_NORM)

        #Optimize
        optimizer.step()
        # scheduler.step(loss)
        scheduler.step()

        backward_done = time.time()
        print("Backward Time: ", backward_done - forward_done)

        # Save Best Epoch
        loss_list.append(loss.item())
        lr_list.append(scheduler.get_last_lr())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_inputs = [ fH_in.detach().cpu(), nH_in.detach().cpu() ]
            best_pred = [CollisionType(omega_vals.H_H.detach().cpu(), omega_vals.H_P.detach().cpu(), omega_vals.H_H2.detach().cpu())]
            best_epoch = epoch


        epoch_runtime = time.time() - epoch_start

        print(
            f"epoch: {epoch:<5} | "
            f"runtime: {epoch_runtime:<8.2} | "
            f"loss: {loss.item():<10.6e} | "
            f"learning rate: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Update Gif data
        if GENERATE_GIF:
            if OPTIMIZE_FH:
                fh_gifgen.update(fH_in[5,9,:], epoch)
            if OPTIMIZE_NH:
                nh_gifgen.update(nH_in, epoch)

    optimization_runtime = time.time() - optim_start
    print(f"Total Optimization Time: {timedelta(seconds=round(optimization_runtime))}")






    # --- Analysis ---

    opt_fH_in, opt_nH_in = best_inputs[0], best_inputs[1]
    omega_vals = best_pred[0]

    # --- Analyze ---
    print("Best Epoch: ", best_epoch)

    # Optimized Inputs Analysis
    if OPTIMIZE_FH:
        fH_in_loss = loss_fun(opt_fH_in, truein_fH).item()
        print("fH Input Loss: ", fH_in_loss)
        print("fH Input Relative L2: ", rel_L2_torch(opt_fH_in, truein_fH))
        print()

    if OPTIMIZE_NH:
        nH_in_loss = loss_fun(opt_nH_in, truein_nH).item()
        print("nH Input Loss: ", nH_in_loss)
        print("nH Input Relative L2: ", rel_L2_torch(opt_nH_in, truein_nH))
        print()


    # Outputs Analysis

    # omega_vals
    OH_H_loss = loss_fun(omega_vals.H_H, trueout_OH_H).item()
    OH_P_loss = loss_fun(omega_vals.H_P, trueout_OH_P).item()
    OH_H2_loss = loss_fun(omega_vals.H_H2, trueout_OH_H2).item()
    print("Omega Loss: ", OH_H_loss, OH_P_loss, OH_H2_loss)
    OH_H_l2 = rel_L2_torch(omega_vals.H_H, trueout_OH_H)
    OH_P_l2 = rel_L2_torch(omega_vals.H_P, trueout_OH_P)
    OH_H2_l2 = rel_L2_torch(omega_vals.H_H2, trueout_OH_H2)
    print("Omega Relative L2: ", OH_H_l2, OH_P_l2, OH_H2_l2)
    print()


    # --- Plot Generation --- 

    print("Generating Images and Gifs")

    # Runtime Data
    generate_loss_plot(image_dir, "Loss", loss_list, xlabel="Epoch", ylabel="Symmetrical Loss")
    generate_lr_plot(image_dir, "LR", lr_list, xlabel="Epoch", ylabel="Learning Rate")

    # fH
    if OPTIMIZE_FH:
        x = range(opt_fH_in.size(2))
        for i in range(opt_fH_in.size(1)):
            generate_compare_plot(image_dir+"fH/", "fH"+str(i), x, opt_fH_in[5,i,:], x, truein_fH[5,i,:], init_x=x, init_y=initial_fH[5,i,:])

    # fH
    if OPTIMIZE_NH:
        x = range(opt_nH_in.numel())
        generate_compare_plot(image_dir+"nH/", "nH", x, opt_nH_in, x, truein_nH, init_x=x, init_y=initial_nH)


    # Outputs
    x = range(omega_vals.H_H.numel())
    generate_compare_plot(image_dir+"Omega/", "H_H", x, omega_vals.H_H, x, trueout_OH_H)
    generate_compare_plot(image_dir+"Omega/", "H_P", x, omega_vals.H_P, x, trueout_OH_P)
    generate_compare_plot(image_dir+"Omega/", "H_H2", x, omega_vals.H_H2, x, trueout_OH_H2)

    # --- Gif Generation ---
    if GENERATE_GIF:
        if OPTIMIZE_FH:
            fh_gifgen.generate_gif()
        if OPTIMIZE_NH:
            nh_gifgen.generate_gif()