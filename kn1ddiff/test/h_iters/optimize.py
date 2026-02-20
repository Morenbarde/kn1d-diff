import torch
import numpy as np
import json
import time
from datetime import timedelta
import math

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.kinetic_h import *
from kn1ddiff.test.utils import *



dir = "kn1ddiff/test/h_iters/"
image_dir = dir+"Images/"
in_file = "kh_iters_in.json"
out_file = "kh_iters_out.json"

# Torch
dtype = torch.float64
USE_CPU = True

# Constants
EPSILON = 10e-10

# Optimization Choices
OPTIMIZE_FH = True
OPTIMIZE_NH = True
OPTIMIZE_GAMMA_WALL = False

# Iteration Parameters
NUM_ITERS = 5
CLIP_NORM = 1e-0

# Learning Rate Parameters
INITIAL_LR = 1e-2
LR_CYCLE_COUNT = 0.2
LR_CYCLE = math.ceil(NUM_ITERS // LR_CYCLE_COUNT)
MIN_LR = 1e-5

# Gif parameters
GENERATE_GIF = True
GIF_FPS = 10
GIF_FREQ = 1


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and not USE_CPU else "cpu")
    print("device: ", device)
    # if use_cuda:
    #     torch.cuda.manual_seed(72)

    # torch.autograd.set_detect_anomaly(True)


    # --- Load Inputs and Outputs ---
    with open(dir+in_file, 'r') as f:
        in_data = json.load(f)
        for key, value in in_data.items():
            in_data[key] = torch.tensor(value, dtype=dtype, device=device)
    with open(dir+out_file, 'r') as f:
        out_data = json.load(f)
        for key, value in out_data.items():
            out_data[key] = torch.tensor(value, dtype=dtype, device=device)

    # Fixed

    # Gradient
    truein_fH = in_data["fH"]
    print("fH Range: ", torch.max(truein_fH), torch.min(truein_fH))

    truein_nH = in_data["nH"]
    print("nH Range: ", torch.max(truein_nH), torch.min(truein_nH))

    truein_gamma_wall = in_data["gamma_wall"]
    print("gamma_wall Range: ", torch.max(truein_gamma_wall), torch.min(truein_gamma_wall))
    # input()


    # Desired Outputs
    
    trueout_fH = out_data["fH"]
    trueout_nH = out_data["nH"]
    trueout_alpha_c = out_data["alpha_c"]
    trueout_Beta_CX_sum = out_data["Beta_CX_sum"]
    trueout_CF_H_H = out_data["CF_H_H"]
    trueout_CF_H_P = out_data["CF_H_P"]
    trueout_CF_H_H2 = out_data["CF_H_H2"]
    trueout_Msum_H_H = out_data["Msum_H_H"]
    trueout_Msum_H_P = out_data["Msum_H_P"]
    trueout_Msum_H_H2 = out_data["Msum_H_H2"]


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
    kinetic_h.H2_Moments.TH2 = in_data['TH2_Moment']
    kinetic_h.H2_Moments.VxH2 = in_data['VxH2_Moment']
    kinetic_h.Internal.Sn = in_data['Sn']
    kinetic_h.Internal.fi_hat = in_data['fi_hat']
    kinetic_h.Internal.Alpha_CX = in_data['Alpha_CX']
    kinetic_h.Internal.alpha_ion = in_data['alpha_ion']
    kinetic_h.Internal.ni = in_data['ni']
    kinetic_h.Internal.SIG_CX = in_data['SIG_CX']
    kinetic_h.Internal.Alpha_H_H2 = in_data['Alpha_H_H2']
    kinetic_h.Internal.Alpha_H_P = in_data['Alpha_H_P']
    kinetic_h.Internal.MH_H_sum = in_data['MH_H_sum']


    # --- Test Input Data ---

    # fH, nH, alpha_c, Beta_CX_sum, collision_freqs, m_sums = kinetic_h._run_iteration_scheme(truein_fH, truein_nH, truein_gamma_wall)
    # check_close("fH", fH, trueout_fH)
    # check_close("nH", nH, trueout_nH)
    # check_close("alpha_c", alpha_c, trueout_alpha_c)
    # check_close("Beta_CX_sum", Beta_CX_sum, trueout_Beta_CX_sum)
    # check_close("CF_H_H", collision_freqs.H_H, trueout_CF_H_H)
    # check_close("CF_H_P", collision_freqs.H_P, trueout_CF_H_P)
    # check_close("CF_H_H2", collision_freqs.H_H2, trueout_CF_H_H2)
    # check_close("Msum_H_H", m_sums.H_H, trueout_Msum_H_H)
    # check_close("Msum_H_P", m_sums.H_P, trueout_Msum_H_P)
    # check_close("Msum_H_H2", m_sums.H_H2, trueout_Msum_H_H2)
    # input()


    # --- Optimization Parameters ---

    initial_fH = 1.1*torch.clone(truein_fH.detach())
    fH_param = torch.nn.Parameter(torch.log(torch.abs(initial_fH)))

    initial_nH = 1.1*torch.clone(truein_nH.detach())
    nH_param = torch.nn.Parameter(torch.log(torch.abs(initial_nH)))

    initial_gamma_wall = 1.05*torch.clone(truein_gamma_wall.detach())
    gamma_wall_param = torch.nn.Parameter(torch.log(torch.abs(initial_gamma_wall)))

    parameters = []
    if OPTIMIZE_FH:
        parameters.append(fH_param)
    if OPTIMIZE_NH:
        parameters.extend([nH_param])
    if OPTIMIZE_GAMMA_WALL:
        parameters.extend([gamma_wall_param])

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
    # loss_fun = lambda pred, true : rel_L2_torch(pred, true)
    # loss_fun = lambda pred, true : torch.mean(
    #         (torch.log(torch.abs(pred) + 1e-12)
    #     - torch.log(torch.abs(true) + 1e-12))**2
    #     )
    # loss_fun = lambda pred, true : ((torch.log(pred + EPSILON) - torch.log(true + EPSILON))**2).mean()

    def symmetric_log(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    loss_fun = lambda pred, true : ((symmetric_log(pred) - symmetric_log(true))**2).mean()










    # --- Optimization ---

    # Init Gif Generator
    if GENERATE_GIF:
        if OPTIMIZE_FH:
            fh_gifgen = GIF_Generator(NUM_ITERS, image_dir+"fH/", "fH", truein_fH[:,5,0], fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_NH:
            nH_gifgen = GIF_Generator(NUM_ITERS, image_dir+"nH/", "nH", truein_nH, fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_GAMMA_WALL:
            gamma_wall_gifgen = GIF_Generator(NUM_ITERS, image_dir+"gamma_wall/", "gamma_wall", truein_gamma_wall, fps=GIF_FPS, frequency=GIF_FREQ)


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

        if OPTIMIZE_GAMMA_WALL:
            gamma_wall_in = torch.exp(gamma_wall_param)
        else:
            gamma_wall_in = truein_gamma_wall


        # --- Run Function ---
        fH_out, nH_out, alpha_c, Beta_CX_sum, coll_freqs, m_sums = kinetic_h._run_iteration_scheme(fH_in, nH_in, gamma_wall_in)

        forward_done = time.time()
        print("Forward Time: ", forward_done - epoch_start)


        # --- Optimize ---

        # Compute Loss
        loss1 = loss_fun(fH_out, trueout_fH)
        loss2 = loss_fun(nH_out, trueout_nH)
        loss3 = loss_fun(alpha_c, trueout_alpha_c)
        loss4 = loss_fun(Beta_CX_sum, trueout_Beta_CX_sum)
        loss5 = loss_fun(coll_freqs.H_H, trueout_CF_H_H)
        loss6 = loss_fun(coll_freqs.H_P, trueout_CF_H_P)
        loss7 = loss_fun(coll_freqs.H_H2, trueout_CF_H_H2)
        loss8 = loss_fun(m_sums.H_H, trueout_Msum_H_H)
        loss9 = loss_fun(m_sums.H_P, trueout_Msum_H_P)
        loss10 = loss_fun(m_sums.H_H2, trueout_Msum_H_H2)
        
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10

        # Backprop
        optimizer.zero_grad()
        loss.backward()

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
            best_inputs = [
                        fH_in.detach(),
                        nH_in.detach(),
                        gamma_wall_in.detach(),
                        ]
            
            best_pred = [
                        fH_out.detach(),
                        nH_out.detach(),
                        alpha_c.detach(),
                        Beta_CX_sum.detach(),
                        CollisionType(
                            coll_freqs.H_H.detach(),
                            coll_freqs.H_P.detach(),
                            coll_freqs.H_H2.detach()
                            ),
                        CollisionType(
                            m_sums.H_H.detach(),
                            m_sums.H_P.detach(),
                            m_sums.H_H2.detach()
                            )
                        ]
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
                fh_gifgen.update(fH_in[:,5,0], epoch)
            if OPTIMIZE_NH:
                nH_gifgen.update(nH_in, epoch)
            if OPTIMIZE_GAMMA_WALL:
                gamma_wall_gifgen.update(gamma_wall_in, epoch)

    optimization_runtime = time.time() - optim_start
    print(f"Total Optimization Time: {timedelta(seconds=round(optimization_runtime))}")






    # --- Analysis ---

    opt_fH_in, opt_nH_in, opt_gamma_wall = best_inputs[0], best_inputs[1], best_inputs[2]
    opt_fH_out, opt_nH_out, opt_alpha_c, opt_Beta_CX_sum, opt_coll_freqs, opt_m_sums = best_pred[0], best_pred[1], best_pred[2], best_pred[3], best_pred[4], best_pred[5]

    # --- Analyze ---
    print("Best Epoch: ", best_epoch)

    # Optimized Inputs Analysis
    if OPTIMIZE_FH:
        analyze_difference("fH Input", loss_fun, opt_fH_in, truein_fH)
        print()

    if OPTIMIZE_NH:
        analyze_difference("nH Input", loss_fun, opt_nH_in, truein_nH)
        print()

    if OPTIMIZE_GAMMA_WALL:
        analyze_difference("gamma wall Input", loss_fun, opt_gamma_wall, truein_gamma_wall)
        print()


    # Outputs Analysis

    #fH
    analyze_difference("fH Output", loss_fun, opt_fH_out, trueout_fH)
    print()

    #nH
    analyze_difference("nH Output", loss_fun, opt_nH_out, trueout_nH)
    print()

    #alpha_c
    analyze_difference("alpha_c Output", loss_fun, opt_alpha_c, trueout_alpha_c)
    print()

    #beta_cx_sum
    analyze_difference("beta_cx_sum Output", loss_fun, opt_Beta_CX_sum, trueout_Beta_CX_sum)
    print()

    #coll_freqs
    analyze_difference("CF_H_H Output", loss_fun, opt_coll_freqs.H_H, trueout_CF_H_H)
    analyze_difference("CF_H_P Output", loss_fun, opt_coll_freqs.H_P, trueout_CF_H_P)
    analyze_difference("CF_H_H2 Output", loss_fun, opt_coll_freqs.H_H2, trueout_CF_H_H2)
    print()

    #m_sums
    analyze_difference("msum_H_H Output", loss_fun, opt_m_sums.H_H, trueout_Msum_H_H)
    analyze_difference("msum_H_P Output", loss_fun, opt_m_sums.H_P, trueout_Msum_H_P)
    analyze_difference("msum_H_H2 Output", loss_fun, opt_m_sums.H_H2, trueout_Msum_H_H2)
    print()


    # --- Plot Generation --- 

    print("Generating Images and Gifs")

    # Runtime Data
    generate_loss_plot(image_dir, "Loss", loss_list, xlabel="Epoch", ylabel="Symmetrical Loss")
    generate_lr_plot(image_dir, "LR", lr_list, xlabel="Epoch", ylabel="Learning Rate")

    # fH
    if OPTIMIZE_FH:
        x = range(opt_fH_in[:,10,0].numel())
        for i in range(len(opt_fH_in[0,:,0])):
            generate_compare_plot(image_dir+"fH/", "fH"+str(i), x, opt_fH_in[:,i,0], x, truein_fH[:,i,0], init_x=x, init_y=initial_fH[:,i,0])

    # nH
    if OPTIMIZE_NH:
        x = range(opt_nH_out.numel())
        generate_compare_plot(image_dir+"nH/", "nH"+str(i), x, opt_nH_in, x, truein_nH, init_x=x, init_y=initial_nH)

    # Gamma Wall
    if OPTIMIZE_GAMMA_WALL:
        x = range(opt_gamma_wall.numel())
        generate_compare_plot(image_dir+"gamma_wall/", "gamma_wall", x, opt_gamma_wall, x, truein_gamma_wall, init_x=x, init_y=initial_gamma_wall)

    # --- Gif Generation ---
    if GENERATE_GIF:
        if OPTIMIZE_FH:
            fh_gifgen.generate_gif()
        if OPTIMIZE_NH:
            nH_gifgen.generate_gif()
        if OPTIMIZE_GAMMA_WALL:
            gamma_wall_gifgen.generate_gif()