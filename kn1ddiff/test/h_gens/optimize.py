import torch
import numpy as np
import json
import time
from datetime import timedelta
import math

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.kinetic_h import *
from kn1ddiff.test.utils import *



dir = "kn1ddiff/test/h_gens/"
image_dir = dir+"Images/"
in_file = "kh_gens_in.json"
out_file = "kh_gens_out.json"

# Torch
dtype = torch.float64
USE_CPU = True

# Constants
EPSILON = 10e-10

# Optimization Choices
OPTIMIZE_FH = True
OPTIMIZE_ALPHA_C = True
OPTIMIZE_COLLISION = True

# Iteration Parameters
NUM_ITERS = 10
CLIP_NORM = 1e-0

# Learning Rate Parameters
INITIAL_LR = 5e-2
LR_CYCLE_COUNT = 1
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
    truein_alpha_c = in_data["alpha_c"]

    print("alpha_c Range: ", torch.max(truein_alpha_c), torch.min(truein_alpha_c))
    truein_CF_H_H = in_data["CF_H_H"]

    print("CF_H_H Range: ", torch.max(truein_CF_H_H), torch.min(truein_CF_H_H))
    truein_CF_H_P = in_data["CF_H_P"]
    print("CF_H_P Range: ", torch.max(truein_CF_H_P), torch.min(truein_CF_H_P))
    truein_CF_H_H2 = in_data["CF_H_H2"]
    print("CF_H_H2 Range: ", torch.max(truein_CF_H_H2), torch.min(truein_CF_H_H2))
    truein_collision_freqs = CollisionType(truein_CF_H_H, truein_CF_H_P, truein_CF_H_H2)
    # input()


    # Desired Outputs
    
    trueout_fH = out_data["fH"]
    trueout_Beta_CX_sum = out_data["Beta_CX_sum"]
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
    kinetic_h.Internal.ni = in_data['ni']
    kinetic_h.Internal.SIG_CX = in_data['SIG_CX']


    # --- Test Input Data ---

    # fH, Beta_CX_sum, m_sums = kinetic_h._run_generations(truein_fH, truein_alpha_c, truein_collision_freqs)
    # print("fH close: ", torch.allclose(fH, trueout_fH))
    # print(rel_L2_torch(fH, trueout_fH))
    # print("Beta_CX_sum close: ", torch.allclose(Beta_CX_sum, trueout_Beta_CX_sum))
    # print(rel_L2_torch(Beta_CX_sum, trueout_Beta_CX_sum))
    # print("Msum_H_H close: ", torch.allclose(m_sums.H_H, trueout_Msum_H_H))
    # print(rel_L2_torch(m_sums.H_H, trueout_Msum_H_H))
    # print("Msum_H_P close: ", torch.allclose(m_sums.H_P, trueout_Msum_H_P))
    # print(rel_L2_torch(m_sums.H_P, trueout_Msum_H_P))
    # print("Msum_H_H2 close: ", torch.allclose(m_sums.H_H2, trueout_Msum_H_H2))
    # print(rel_L2_torch(m_sums.H_H2, trueout_Msum_H_H2))
    # input()


    # --- Optimization Parameters ---

    initial_fH = 1.05*torch.clone(truein_fH.detach())
    fH_param = torch.nn.Parameter(torch.log(torch.abs(initial_fH)))

    initial_alpha_c = 1.05*torch.clone(truein_alpha_c.detach())
    alpha_c_param = torch.nn.Parameter(torch.log(torch.abs(initial_alpha_c)))

    initial_CF_H_H = 1.05*torch.clone(truein_CF_H_H.detach())
    CF_H_H_param = torch.nn.Parameter(torch.log(torch.abs(initial_CF_H_H)))
    initial_CF_H_P = 1.05*torch.clone(truein_CF_H_P.detach())
    CF_H_P_param = torch.nn.Parameter(torch.log(torch.abs(initial_CF_H_P)))
    initial_CF_H_H2 = 1.05*torch.clone(truein_CF_H_H2.detach())
    CF_H_H2_param = torch.nn.Parameter(torch.log(torch.abs(initial_CF_H_H2)))

    parameters = []
    if OPTIMIZE_FH:
        parameters.append(fH_param)
    if OPTIMIZE_ALPHA_C:
        parameters.extend([alpha_c_param])
    if OPTIMIZE_COLLISION:
        parameters.extend([CF_H_H_param, CF_H_P_param, CF_H_H2_param])

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
            fh_gifgen = GIF_Generator(NUM_ITERS, image_dir+"fH/", "fH", truein_fH[:,10,0], fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_ALPHA_C:
            alpha_c_gifgen = GIF_Generator(NUM_ITERS, image_dir+"alpha_c/", "alpha_c", truein_alpha_c[5,10,:], fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_COLLISION:
            H_H_gifgen = GIF_Generator(NUM_ITERS, image_dir+"Collision_Frequency/", "H_H", truein_CF_H_H, fps=GIF_FPS, frequency=GIF_FREQ)
            H_P_gifgen = GIF_Generator(NUM_ITERS, image_dir+"Collision_Frequency/", "H_P", truein_CF_H_P, fps=GIF_FPS, frequency=GIF_FREQ)
            H_H2_gifgen = GIF_Generator(NUM_ITERS, image_dir+"Collision_Frequency/", "H_H2", truein_CF_H_H2, fps=GIF_FPS, frequency=GIF_FREQ)


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

        if OPTIMIZE_ALPHA_C:
            alpha_c = torch.exp(alpha_c_param)
        else:
            alpha_c = truein_alpha_c

        if OPTIMIZE_COLLISION:
            # cf_hh = 1e-2 * torch.sigmoid(initial_CF_H_H)
            cf_hh = torch.exp(CF_H_H_param)
            # cf_hp = 1e+1 * torch.sigmoid(initial_CF_H_P)
            cf_hp = torch.exp(CF_H_P_param)
            # cf_hh2 = 1e-1 * torch.sigmoid(initial_CF_H_H2)
            cf_hh2 = torch.exp(CF_H_H2_param)
            coll_freqs = CollisionType(cf_hh, cf_hp, cf_hh2)
        else:
            coll_freqs = truein_collision_freqs


        # --- Run Function ---
        fH_out, Beta_CX_sum, m_sums = kinetic_h._run_generations(fH_in, alpha_c, coll_freqs)

        forward_done = time.time()
        print("Forward Time: ", forward_done - epoch_start)


        # --- Optimize ---

        # Compute Loss
        loss1 = loss_fun(fH_out, trueout_fH)
        loss2 = loss_fun(Beta_CX_sum, trueout_Beta_CX_sum)
        loss3 = loss_fun(m_sums.H_H, trueout_Msum_H_H)
        loss4 = loss_fun(m_sums.H_P, trueout_Msum_H_P)
        loss5 = loss_fun(m_sums.H_H2, trueout_Msum_H_H2)
        
        loss = loss1 + loss2 + loss3 + loss4 + loss5

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
                        fH_in.detach().cpu(),
                        alpha_c.detach().cpu(),
                        CollisionType(
                            coll_freqs.H_H.detach().cpu(),
                            coll_freqs.H_P.detach().cpu(),
                            coll_freqs.H_H2.detach().cpu()
                            )
                        ]
            
            best_pred = [fH_out.detach().cpu(), Beta_CX_sum.detach().cpu(), CollisionType(m_sums.H_H.detach().cpu(), m_sums.H_P.detach().cpu(), m_sums.H_H2.detach().cpu())]
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
                fh_gifgen.update(fH_in[:,10,0], epoch)
            if OPTIMIZE_ALPHA_C:
                alpha_c_gifgen.update(alpha_c[5,10,:], epoch)
            if OPTIMIZE_COLLISION:
                H_H_gifgen.update(cf_hh, epoch)
                H_P_gifgen.update(cf_hp, epoch)
                H_H2_gifgen.update(cf_hh2, epoch)

    optimization_runtime = time.time() - optim_start
    print(f"Total Optimization Time: {timedelta(seconds=round(optimization_runtime))}")






    # --- Analysis ---

    opt_fH_in, opt_alpha_c, opt_coll_freq = best_inputs[0], best_inputs[1], best_inputs[2]
    opt_fH_out, opt_Beta_CX_sum, opt_m_sums = best_pred[0], best_pred[1], best_pred[2]

    # --- Analyze ---
    print("Best Epoch: ", best_epoch)

    # Optimized Inputs Analysis
    if OPTIMIZE_FH:
        fH_in_loss = loss_fun(opt_fH_in, truein_fH).item()
        print("fH Input Loss: ", fH_in_loss)
        print("fH Input Relative L2: ", rel_L2_torch(opt_fH_in, truein_fH))
        print()

    if OPTIMIZE_ALPHA_C:
        alpha_c_loss = loss_fun(opt_alpha_c, truein_alpha_c).item()
        print("alpha_c Loss: ", alpha_c_loss)
        alpha_c_l2 = rel_L2_torch(opt_alpha_c, truein_alpha_c)
        print("alpha_c Relative L2: ", alpha_c_l2)
        print()

    if OPTIMIZE_COLLISION:
        CF_H_H_loss = loss_fun(opt_coll_freq.H_H, truein_CF_H_H).item()
        CF_H_P_loss = loss_fun(opt_coll_freq.H_P, truein_CF_H_P).item()
        CF_H_H2_loss = loss_fun(opt_coll_freq.H_H2, truein_CF_H_H2).item()
        print("Collision Loss: ", CF_H_H_loss, CF_H_P_loss, CF_H_H2_loss)
        CF_H_H_l2 = rel_L2_torch(opt_coll_freq.H_H, truein_CF_H_H)
        CF_H_P_l2 = rel_L2_torch(opt_coll_freq.H_P, truein_CF_H_P)
        CF_H_H2_l2 = rel_L2_torch(opt_coll_freq.H_H2, truein_CF_H_H2)
        print("Collision L2: ", CF_H_H_l2, CF_H_P_l2, CF_H_H2_l2)
        print()

    # Outputs Analysis

    #fH
    fH_out_loss = loss_fun(opt_fH_out, trueout_fH).item()
    print("fH Output Loss: ", fH_out_loss)
    print("fH Output Relative L2: ", rel_L2_torch(opt_fH_out, trueout_fH))
    print()

    #beta_cx_sum
    beta_cx_sum_loss = loss_fun(opt_Beta_CX_sum, trueout_Beta_CX_sum).item()
    print("Beta_CX_Sum Loss: ", beta_cx_sum_loss)
    print("Beta_CX_Sum Relative L2: ", rel_L2_torch(opt_Beta_CX_sum, trueout_Beta_CX_sum))
    print()

    #m_sums
    msum_H_H_loss = loss_fun(opt_m_sums.H_H, trueout_Msum_H_H).item()
    msum_H_P_loss = loss_fun(opt_m_sums.H_P, trueout_Msum_H_P).item()
    msum_H_H2_loss = loss_fun(opt_m_sums.H_H2, trueout_Msum_H_H2).item()
    print("M_Sum Loss: ", msum_H_H_loss, msum_H_P_loss, msum_H_H2_loss)
    msum_H_H_l2 = rel_L2_torch(opt_m_sums.H_H, trueout_Msum_H_H)
    msum_H_P_l2 = rel_L2_torch(opt_m_sums.H_P, trueout_Msum_H_P)
    msum_H_H2_l2 = rel_L2_torch(opt_m_sums.H_H2, trueout_Msum_H_H2)
    print("M_Sum Relative L2: ", msum_H_H_l2, msum_H_P_l2, msum_H_H2_l2)
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

    # MEQ Coeffs
    if OPTIMIZE_ALPHA_C:
        x = range(opt_alpha_c[5,10,:].numel())
        for i in range(len(opt_alpha_c[5,:,0])):
            generate_compare_plot(image_dir+"alpha_c/", "alpha_c"+str(i), x, opt_alpha_c[5,i,:], x, truein_alpha_c[5,i,:], init_x=x, init_y=initial_alpha_c[5,i,:])

    # Collision Frequencies
    if OPTIMIZE_COLLISION:
        x = range(opt_coll_freq.H_H.numel())
        generate_compare_plot(image_dir+"Collision_Frequency/", "H_H", x, opt_coll_freq.H_H, x, truein_CF_H_H, init_x=x, init_y=initial_CF_H_H)
        generate_compare_plot(image_dir+"Collision_Frequency/", "H_P", x, opt_coll_freq.H_P, x, truein_CF_H_P, init_x=x, init_y=initial_CF_H_P)
        generate_compare_plot(image_dir+"Collision_Frequency/", "H_H2", x, opt_coll_freq.H_H2, x, truein_CF_H_H2, init_x=x, init_y=initial_CF_H_H2)

    # --- Gif Generation ---
    if GENERATE_GIF:
        if OPTIMIZE_FH:
            fh_gifgen.generate_gif()
        if OPTIMIZE_ALPHA_C:
            alpha_c_gifgen.generate_gif()
        if OPTIMIZE_COLLISION:
            H_H_gifgen.generate_gif()
            H_P_gifgen.generate_gif()
            H_H2_gifgen.generate_gif()