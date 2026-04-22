import torch
import numpy as np
import json
import time
from datetime import timedelta, datetime
import math
from dataclasses import asdict
from scipy.io import readsav

from kn1ddiff.kn1d import kn1d, KN1DResults
from Optimization.utils import *


# File Output
USE_LOG = True

# Torch
dtype = torch.float64
USE_CPU = True

# Constants
EPSILON = 10e-10

# Optimization Choices
OPTIMIZE_NE = True
OPTIMIZE_TI = True
OPTIMIZE_TE = True
OPTIMIZE_GAUGEH2 = True

# Factors for setting initial values for optimization. 
INIT_FACTOR = 0.9 # initial variables are multiplied by this factor as a starting point
OFFSET_FACTOR = 0.0 # initial variables are offset by themselves times this factor as a starting point
LOSS_FUNC = "sym" # "log" or "sym"

# Iteration Parameters
NUM_ITERS = 1000
NUM_THREADS = 2
CLIP_NORM = 1e-0

# Learning Rate Parameters
INITIAL_LR = 1e-2
CYCLE_LR = False
LR_CYCLE_COUNT = 1
LR_CYCLE = math.ceil(NUM_ITERS // LR_CYCLE_COUNT)
MIN_LR = 1e-6

# Gif parameters
GENERATE_GIF = True
# GENERATE_GRADIENT_GIF = True
GIF_FPS = 5
GIF_FREQ = 50

# Folder Settings
now = datetime.now()
exp = np.floor(np.log10(np.abs(INITIAL_LR)))
base = int(INITIAL_LR*(10**(-exp)))
exp = int(exp)
folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")+"_"+str(NUM_ITERS)+f'_{base}e{exp}'+"_Cycle-"+str(CYCLE_LR)

local_dir = "Optimization/kn1d/"
run_dir = local_dir+"Runs/"+folder_name+"/"
image_dir = run_dir+"Images/"
data_dir = run_dir+"Data/"
# in_file = "kn1d_in.json"
out_file = "kn1d_out.json"


if __name__ == "__main__":

    # Start logging to run file
    if USE_LOG:
        setup_log(run_dir)

    print("Process PID: ", os.getpid())


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and not USE_CPU else "cpu")
    print("device: ", device)
    # if use_cuda:
    #     torch.cuda.manual_seed(72)


    ### Print Run Info ###

    print()
    print("Optimizing: ")
    print("    N: ", OPTIMIZE_NE)
    print("    Te: ", OPTIMIZE_TE)
    print("    Ti: ", OPTIMIZE_TI)
    print("    GaugeH2: ", OPTIMIZE_GAUGEH2)
    print()
    print("Thread Count: ", NUM_THREADS)
    print("Iteration Count: ", NUM_ITERS)
    print("Initial Offset: *"+str(INIT_FACTOR)+", +"+str(OFFSET_FACTOR))
    print("Initial Learning Rate: ", INITIAL_LR)
    if CYCLE_LR:
        print("Min Learning Rate: ", MIN_LR)
        print("Learning Rate Cycle Length: ", LR_CYCLE)
    print("Gradient Clipping: ", CLIP_NORM)
    print("Loss Function: ", LOSS_FUNC)
    print()

    # torch.autograd.set_detect_anomaly(True)
    torch.set_num_threads(NUM_THREADS)
    # torch._dynamo.config.capture_scalar_outputs = True

    # --- Load Inputs and Outputs ---

    data_file = './sav_files/kn1d_test_inputs.sav'
    # data_file = './sav_files/1090904018_950to1050.sav'
    # data_file = './sav_files/1090904029_950to1050_towall.sav'
    print("Loading file: "  + data_file)
    in_data = readsav(data_file)
    for key, value in in_data.items():
        in_data[key] = torch.tensor(np.asarray(value).astype(np.float32), dtype=dtype, device=device)

    with open(local_dir+out_file, 'r') as f:
        out_data = json.load(f)
        for key, value in out_data.items():
            out_data[key] = torch.tensor(value, dtype=dtype, device=device)


    
    # Fixed

    # Gradient
    truein_ne = in_data["n_e"]
    print_torch_range(truein_ne, "ne")
    truein_Ti = in_data["t_i"]
    print_torch_range(truein_Ti, "Ti")
    truein_Te = in_data["t_e"]
    print_torch_range(truein_Te, "Te")
    truein_GaugeH2 = in_data["p_wall"]
    print_torch_range(truein_GaugeH2, "GaugeH2")
    # input()


    # Desired Outputs

    fixed_xH = out_data["xH"]
    fixed_xH2 = out_data["xH2"]

    trueout_nH2 = out_data["nH2"]
    trueout_GammaxH2 = out_data["GammaxH2"]
    trueout_TH2 = out_data["TH2"]
    trueout_qxH2_total = out_data["qxH2_total"]
    trueout_nHP = out_data["nHP"]
    trueout_THP = out_data["THP"]
    trueout_SH = out_data["SH"]
    trueout_SP = out_data["SP"]

    trueout_nH = out_data["nH"]
    trueout_GammaxH = out_data["GammaxH"]
    trueout_TH = out_data["TH"]
    trueout_qxH_total = out_data["qxH_total"]
    trueout_NetHSource = out_data["NetHSource"]
    trueout_Sion = out_data["Sion"]
    trueout_QH_total = out_data["QH_total"]
    trueout_SideWallH = out_data["SideWallH"]
    trueout_Lyman = out_data["Lyman"]
    trueout_Balmer = out_data["Balmer"]
    trueout_GammaHLim = out_data["GammaHLim"]



    # --- Test Input Data ---

    kn1d_results = kn1d(in_data['x'], in_data['x_lim'], in_data['x_sep'], truein_GaugeH2, in_data['mu'], truein_Ti, 
               truein_Te, truein_ne, in_data['vx'], in_data['lc'], in_data['d_pipe'],
               xH=fixed_xH, xH2=fixed_xH2, 
               max_gen=100, Hdebug=False, H2debug=False, debrief = False, Hdebrief = False, H2debrief = False, compute_errors = False, save_results=False)

    check_close("nH2", kn1d_results.nH2, trueout_nH2)
    check_close("GammaxH2", kn1d_results.GammaxH2, trueout_GammaxH2)
    check_close("TH2", kn1d_results.TH2, trueout_TH2)
    check_close("qxH2_total", kn1d_results.qxH2_total, trueout_qxH2_total)
    check_close("nHP", kn1d_results.nHP, trueout_nHP)
    check_close("THP", kn1d_results.THP, trueout_THP)
    check_close("SH", kn1d_results.SH, trueout_SH)
    check_close("SP", kn1d_results.SP, trueout_SP)

    check_close("nH", kn1d_results.nH, trueout_nH)
    check_close("GammaxH", kn1d_results.GammaxH, trueout_GammaxH)
    check_close("TH", kn1d_results.TH, trueout_TH)
    check_close("qxH_total", kn1d_results.qxH_total, trueout_qxH_total)
    check_close("NetHSource", kn1d_results.NetHSource, trueout_NetHSource)
    check_close("Sion", kn1d_results.Sion, trueout_Sion)
    check_close("QH_total", kn1d_results.QH_total, trueout_QH_total)
    check_close("SideWallH", kn1d_results.SideWallH, trueout_SideWallH)
    check_close("Lyman", kn1d_results.Lyman, trueout_Lyman)
    check_close("Balmer", kn1d_results.Balmer, trueout_Balmer)
    check_close("GammaHLim", kn1d_results.GammaHLim, trueout_GammaHLim)
    print()
    # input()

    trueout_results = kn1d_results


    # --- Optimization Parameters ---
    
    parameterize = lambda tensor : torch.nn.Parameter(torch.log(torch.abs(tensor)))

    initial_ne = init_optimization_tensor(truein_ne, INIT_FACTOR, OFFSET_FACTOR)
    ne_param = parameterize(initial_ne)
    initial_Ti = init_optimization_tensor(truein_Ti, INIT_FACTOR, OFFSET_FACTOR)
    Ti_param = parameterize(initial_Ti)
    initial_Te = init_optimization_tensor(truein_Te, INIT_FACTOR, OFFSET_FACTOR)
    Te_param = parameterize(initial_Te)
    initial_GaugeH2 = init_optimization_tensor(truein_GaugeH2, INIT_FACTOR, OFFSET_FACTOR)
    GaugeH2_param = parameterize(initial_GaugeH2)


    parameters = []

    if OPTIMIZE_NE:
        parameters.append(ne_param)
    if OPTIMIZE_TI:
        parameters.extend([Ti_param])
    if OPTIMIZE_TE:
        parameters.extend([Te_param])
    if OPTIMIZE_GAUGEH2:
        parameters.extend([GaugeH2_param])


    optimizer = torch.optim.Adam(parameters, lr=INITIAL_LR, betas=(0.9, 0.999))


    # --- Scheduler Options --- 

    print("Learning Rate Cycle: ", LR_CYCLE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=LR_CYCLE,
        # T_mult=1,
        eta_min=MIN_LR,
    )


    # --- Loss Function Options --- 

    if LOSS_FUNC == "sym":
        def symmetric_log(x):
            return torch.sign(x) * torch.log1p(torch.abs(x))
        loss_fun = lambda pred, true : ((symmetric_log(pred) - symmetric_log(true))**2).mean()

    elif LOSS_FUNC == "log":
        loss_fun = lambda pred, true : torch.log1p((pred-true)**2).mean()








    # --- Optimization ---

    # Init Gif Generator
    if GENERATE_GIF:
        if OPTIMIZE_NE:
            ne_gifgen = GIF_Generator(NUM_ITERS, image_dir, "ne", truein_ne, fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_TI:
            Ti_gifgen = GIF_Generator(NUM_ITERS, image_dir, "Ti", truein_Ti, fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_TE:
            Te_gifgen = GIF_Generator(NUM_ITERS, image_dir, "Te", truein_Te, fps=GIF_FPS, frequency=GIF_FREQ)
        # if OPTIMIZE_GAUGEH2:
        #     GaugeH2_gifgen = GIF_Generator(NUM_ITERS, image_dir, "GaugeH2", truein_GaugeH2, fps=GIF_FPS, frequency=GIF_FREQ)

    # if GENERATE_GRADIENT_GIF:
    #     if OPTIMIZE_NE:
    #         ne_grad_gifgen = GIF_Generator(NUM_ITERS, image_dir, "ne_grad", fps=GIF_FPS, frequency=GIF_FREQ)
    #     if OPTIMIZE_TI:
    #         Ti_grad_gifgen = GIF_Generator(NUM_ITERS, image_dir, "Ti_grad", fps=GIF_FPS, frequency=GIF_FREQ)
    #     if OPTIMIZE_TE:
    #         Te_grad_gifgen = GIF_Generator(NUM_ITERS, image_dir, "Te_grad", fps=GIF_FPS, frequency=GIF_FREQ)


    # Capture Best Epoch
    loss_list = []
    lr_list = []
    best_loss = np.inf
    best_epoch = 0

    optim_start = time.time()

    for epoch in range(NUM_ITERS):

        epoch_start = time.time()

        # --- Bound Inputs ---


        ne_in = torch.exp(ne_param) if OPTIMIZE_NE else truein_ne
        Ti_in = torch.exp(Ti_param) if OPTIMIZE_TI else truein_Ti
        Te_in = torch.exp(Te_param) if OPTIMIZE_TE else truein_Te
        GaugeH2_in = torch.exp(GaugeH2_param)  if OPTIMIZE_GAUGEH2 else truein_GaugeH2

        # --- Run Function ---
        kn1d_results = kn1d(in_data['x'], in_data['x_lim'], in_data['x_sep'], GaugeH2_in, in_data['mu'], Ti_in, 
                            Te_in, ne_in, in_data['vx'], in_data['lc'], in_data['d_pipe'],
                            xH=fixed_xH, xH2=fixed_xH2, 
                            max_gen=100, Hdebug=False, H2debug=False, debrief = False, Hdebrief = False, H2debrief = False, compute_errors = False, save_results=False)

        forward_done = time.time()
        forward_time = forward_done - epoch_start


        # --- Optimize ---

        # Compute Loss

        loss1 = loss_fun(kn1d_results.nH2, trueout_results.nH2)
        loss2 = loss_fun(kn1d_results.GammaxH2, trueout_results.GammaxH2)
        loss3 = loss_fun(kn1d_results.TH2, trueout_results.TH2)
        loss4 = loss_fun(kn1d_results.qxH2_total, trueout_results.qxH2_total)
        loss5 = loss_fun(kn1d_results.nHP, trueout_results.nHP)
        loss6 = loss_fun(kn1d_results.THP, trueout_results.THP)
        loss7 = loss_fun(kn1d_results.SH, trueout_results.SH)
        loss8 = loss_fun(kn1d_results.SP, trueout_results.SP)

        loss9 = loss_fun(kn1d_results.nH, trueout_results.nH)
        loss10 = loss_fun(kn1d_results.GammaxH, trueout_results.GammaxH)
        loss11 = loss_fun(kn1d_results.TH, trueout_results.TH)
        loss12 = loss_fun(kn1d_results.qxH_total, trueout_results.qxH_total)
        loss13 = loss_fun(kn1d_results.NetHSource, trueout_results.NetHSource)
        loss14 = loss_fun(kn1d_results.Sion, trueout_results.Sion)
        loss15 = loss_fun(kn1d_results.QH_total, trueout_results.QH_total)
        loss16 = loss_fun(kn1d_results.SideWallH, trueout_results.SideWallH)
        loss17 = loss_fun(kn1d_results.Lyman, trueout_results.Lyman)
        loss18 = loss_fun(kn1d_results.Balmer, trueout_results.Balmer)
        loss19 = loss_fun(kn1d_results.GammaHLim, trueout_results.GammaHLim)

        
        loss = (loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10+loss11+loss12+loss13
                +loss14+loss15+loss16+loss17+loss18+loss19)

        # print("Loss1", loss1.item())
        # print("Loss2", loss2.item())
        # print("Loss3", loss3.item())
        # print("Loss4", loss4.item())
        # print("Loss5", loss5.item())
        # print("Loss6", loss6.item())
        # print("Loss7", loss7.item())
        # print("Loss8", loss8.item())
        # print("Loss9", loss9.item())
        # print("Loss10", loss10.item())
        # print("Loss11", loss11.item())
        # print("Loss12", loss12.item())
        # print("Loss13", loss13.item())
        # print("Loss14", loss14.item())
        # print("Loss15", loss15.item())
        # print("Loss16", loss16.item())
        # print("Loss17", loss17.item())
        # print("Loss18", loss18.item())
        # print("Loss19", loss19.item())
        # print("Loss20", loss20.item())
        # print("Loss21", loss21.item())
        # print("Loss22", loss22.item())
        # print("Loss23", loss23.item())
        # print("Loss24", loss24.item())
        # print("Loss Total", loss.item())
        # input()

        # Backprop
        optimizer.zero_grad()
        loss.backward()

        # Clip Gradient
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=CLIP_NORM)


        # if GENERATE_GRADIENT_GIF:
        #     if OPTIMIZE_NE:
        #         ne_grad_gifgen.update(ne_param.grad, epoch)
        #     if OPTIMIZE_TI:
        #         Ti_grad_gifgen.update(Ti_param.grad, epoch)
        #     if OPTIMIZE_TE:
        #         Te_grad_gifgen.update(Te_param.grad, epoch)

        #Optimize
        optimizer.step()
        if CYCLE_LR:
            # scheduler.step(loss)
            scheduler.step()

        backward_done = time.time()
        backward_time = backward_done - forward_done

        # Save Best Epoch
        loss_list.append(loss.item())
        lr_list.append(scheduler.get_last_lr()[0])
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_inputs = {
                            "Ti" : Ti_in.detach().cpu(),
                            "Te" : Te_in.detach().cpu(),
                            "ne" : ne_in.detach().cpu(),
                            "GaugeH2" : GaugeH2_in.detach().cpu(),
                            }
            
            best_pred = KN1DResults(
                                kn1d_results.xH2,
                                kn1d_results.nH2,
                                kn1d_results.GammaxH2,
                                kn1d_results.TH2,
                                kn1d_results.qxH2_total,
                                kn1d_results.nHP,
                                kn1d_results.THP,
                                kn1d_results.SH,
                                kn1d_results.SP,

                                kn1d_results.xH,
                                kn1d_results.nH,
                                kn1d_results.GammaxH,
                                kn1d_results.TH,
                                kn1d_results.qxH_total,
                                kn1d_results.NetHSource,
                                kn1d_results.Sion,
                                kn1d_results.QH_total,
                                kn1d_results.SideWallH,
                                kn1d_results.Lyman,
                                kn1d_results.Balmer,
                                kn1d_results.GammaHLim,
                                )
            best_epoch = epoch


        epoch_runtime = time.time() - epoch_start

        print(
            f"epoch: {epoch:<5} | "
            f"runtime: {epoch_runtime:<5.2f}   F: {forward_time:<5.2f}  B: {backward_time:<5.2f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | "
            f"loss: {loss.item():<10.6e}"
        )

        # Update Gif data
        if GENERATE_GIF:
            if OPTIMIZE_NE:
                ne_gifgen.update(ne_in, epoch)
            if OPTIMIZE_TI:
                Ti_gifgen.update(Ti_in, epoch)
            if OPTIMIZE_TE:
                Te_gifgen.update(Te_in, epoch)

    optimization_runtime = time.time() - optim_start
    print(f"Total Optimization Time: {timedelta(seconds=round(optimization_runtime))}")






    # --- Analysis ---

    opt_inputs = best_inputs
    opt_results = best_pred

    # --- Save Results ---

    check_and_generate_dir(data_dir)

    file = 'opt_in.json'
    print("Saving to file: " + file)
    sav_data = {
        "Ti" : opt_inputs["Ti"],
        "Te" : opt_inputs["Te"],
        "ne" : opt_inputs["ne"],
        "GaugeH2" : opt_inputs["GaugeH2"]
    }
    sav_data = make_json_compatible(sav_data)
    sav_to_json(data_dir+file, sav_data)

    file = 'opt_out.json'
    print("Saving to file: " + file)
    sav_data = {
        "nH2": opt_results.nH2,
        "GammaxH2": opt_results.GammaxH2,
        "TH2": opt_results.TH2,
        "qxH2_total": opt_results.qxH2_total,
        "nHP": opt_results.nHP,
        "THP": opt_results.THP,
        "SH": opt_results.SH,
        "SP": opt_results.SP,

        "nH": opt_results.nH,
        "GammaxH": opt_results.GammaxH,
        "TH": opt_results.TH,
        "qxH_total": opt_results.qxH_total,
        "NetHSource": opt_results.NetHSource,
        "Sion": opt_results.Sion,
        "QH_total": opt_results.QH_total,
        "SideWallH": opt_results.SideWallH,
        "Lyman": opt_results.Lyman,
        "Balmer": opt_results.Balmer,
        "GammaHLim": opt_results.GammaHLim,
    }
    sav_data = make_json_compatible(sav_data)
    sav_to_json(data_dir+file, sav_data)

    # Save true values

    file = 'true_in.json'
    print("Saving to file: " + file)
    sav_data = {
        "Ti" : truein_Ti,
        "Te" : truein_Te,
        "ne" : truein_ne,
        "GaugeH2" : truein_GaugeH2
    }
    sav_data = make_json_compatible(sav_data)
    sav_to_json(data_dir+file, sav_data)

    file = 'true_out.json'
    print("Saving to file: " + file)
    sav_data = {
        "nH2": trueout_results.nH2,
        "GammaxH2": trueout_results.GammaxH2,
        "TH2": trueout_results.TH2,
        "qxH2_total": trueout_results.qxH2_total,
        "nHP": trueout_results.nHP,
        "THP": trueout_results.THP,
        "SH": trueout_results.SH,
        "SP": trueout_results.SP,

        "nH": trueout_results.nH,
        "GammaxH": trueout_results.GammaxH,
        "TH": trueout_results.TH,
        "qxH_total": trueout_results.qxH_total,
        "NetHSource": trueout_results.NetHSource,
        "Sion": trueout_results.Sion,
        "QH_total": trueout_results.QH_total,
        "SideWallH": trueout_results.SideWallH,
        "Lyman": trueout_results.Lyman,
        "Balmer": trueout_results.Balmer,
        "GammaHLim": trueout_results.GammaHLim,
    }
    sav_data = make_json_compatible(sav_data)
    sav_to_json(data_dir+file, sav_data)


    # Save final output

    file = 'final_in.json'
    print("Saving to file: " + file)
    sav_data = {
        "Ti" : Ti_in.detach().cpu(),
        "Te" : Te_in.detach().cpu(),
        "ne" : ne_in.detach().cpu(),
        "GaugeH2" : GaugeH2_in.detach().cpu()
    }
    sav_data = make_json_compatible(sav_data)
    sav_to_json(data_dir+file, sav_data)

    file = 'final_out.json'
    print("Saving to file: " + file)
    sav_data = {
        "nH2": kn1d_results.nH2,
        "GammaxH2": kn1d_results.GammaxH2,
        "TH2": kn1d_results.TH2,
        "qxH2_total": kn1d_results.qxH2_total,
        "nHP": kn1d_results.nHP,
        "THP": kn1d_results.THP,
        "SH": kn1d_results.SH,
        "SP": kn1d_results.SP,

        "nH": kn1d_results.nH,
        "GammaxH": kn1d_results.GammaxH,
        "TH": kn1d_results.TH,
        "qxH_total": kn1d_results.qxH_total,
        "NetHSource": kn1d_results.NetHSource,
        "Sion": kn1d_results.Sion,
        "QH_total": kn1d_results.QH_total,
        "SideWallH": kn1d_results.SideWallH,
        "Lyman": kn1d_results.Lyman,
        "Balmer": kn1d_results.Balmer,
        "GammaHLim": kn1d_results.GammaHLim,
    }
    sav_data = make_json_compatible(sav_data)
    sav_to_json(data_dir+file, sav_data)


    file = 'loss_lr.json'
    print("Saving to file: " + file)
    sav_data = {
        "loss" : loss_list,
        "lr" : lr_list
    }
    sav_data = make_json_compatible(sav_data)
    sav_to_json(data_dir+file, sav_data)



    # --- Analyze ---
    print("Best Epoch: ", best_epoch)

    print("### Inputs Analysis ###")

    # Optimized Inputs Analysis

    if OPTIMIZE_NE:
        analyze_difference("ne", loss_fun, opt_inputs["ne"], truein_ne)
        print()
    if OPTIMIZE_TI:
        analyze_difference("Ti", loss_fun, opt_inputs["Ti"], truein_Ti)
        print()
    if OPTIMIZE_TE:
        analyze_difference("Te", loss_fun, opt_inputs["Te"], truein_Te)
        print()
    if OPTIMIZE_GAUGEH2:
        analyze_difference("GaugeH2", loss_fun, opt_inputs["GaugeH2"], truein_GaugeH2)
        print()

    # Outputs Analysis

    print("### Outputs Analysis ###")


    analyze_difference("nH2", loss_fun, opt_results.nH2, trueout_results.nH2)
    analyze_difference("GammaxH2", loss_fun, opt_results.GammaxH2, trueout_results.GammaxH2)
    analyze_difference("TH2", loss_fun, opt_results.TH2, trueout_results.TH2)
    analyze_difference("qxH2_total", loss_fun, opt_results.qxH2_total, trueout_results.qxH2_total)
    analyze_difference("nHP", loss_fun, opt_results.nHP, trueout_results.nHP)
    analyze_difference("THP", loss_fun, opt_results.THP, trueout_results.THP)
    analyze_difference("SH", loss_fun, opt_results.SH, trueout_results.SH)
    analyze_difference("SP", loss_fun, opt_results.SP, trueout_results.SP)

    analyze_difference("nH", loss_fun, opt_results.nH, trueout_results.nH)
    analyze_difference("GammaxH", loss_fun, opt_results.GammaxH, trueout_results.GammaxH)
    analyze_difference("TH", loss_fun, opt_results.TH, trueout_results.TH)
    analyze_difference("qxH_total", loss_fun, opt_results.qxH_total, trueout_results.qxH_total)
    analyze_difference("NetHSource", loss_fun, opt_results.NetHSource, trueout_results.NetHSource)
    analyze_difference("Sion", loss_fun, opt_results.Sion, trueout_results.Sion)
    analyze_difference("QH_total", loss_fun, opt_results.QH_total, trueout_results.QH_total)
    analyze_difference("SideWallH", loss_fun, opt_results.SideWallH, trueout_results.SideWallH)
    analyze_difference("Lyman", loss_fun, opt_results.Lyman, trueout_results.Lyman)
    analyze_difference("Balmer", loss_fun, opt_results.Balmer, trueout_results.Balmer)
    analyze_difference("GammaHLim", loss_fun, opt_results.GammaHLim, trueout_results.GammaHLim)

    # --- Plot Generation --- 

    print("Generating Images and Gifs")

    # Runtime Data
    generate_loss_plot(image_dir, "Loss", loss_list, xlabel="Epoch", ylabel="Symmetrical Loss")
    generate_lr_plot(image_dir, "LR", lr_list, xlabel="Epoch", ylabel="Learning Rate")
    
    x = in_data['x']
    if OPTIMIZE_NE:
        generate_compare_plot(image_dir, "ne", x, opt_inputs["ne"], x, truein_ne, init_x=x, init_y=initial_ne)
    if OPTIMIZE_TI:
        generate_compare_plot(image_dir, "Ti", x, opt_inputs["Ti"], x, truein_Ti, init_x=x, init_y=initial_Ti)
    if OPTIMIZE_TE:
        generate_compare_plot(image_dir, "Te", x, opt_inputs["Te"], x, truein_Te, init_x=x, init_y=initial_Te)

    # --- Gif Generation ---
    if GENERATE_GIF:
        if OPTIMIZE_NE:
            ne_gifgen.generate_gif()
        if OPTIMIZE_TI:
            Ti_gifgen.generate_gif()
        if OPTIMIZE_TE:
            Te_gifgen.generate_gif()

    # if GENERATE_GRADIENT_GIF:
    #     if OPTIMIZE_NE:
    #         ne_grad_gifgen.generate_gif()
    #     if OPTIMIZE_TI:
    #         Ti_grad_gifgen.generate_gif()
    #     if OPTIMIZE_TE:
    #         Te_grad_gifgen.generate_gif()