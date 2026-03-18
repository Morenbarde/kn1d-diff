import torch
import numpy as np
import json
import time
from datetime import timedelta, datetime
import math
from dataclasses import asdict

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.kinetic_h2 import *
from kn1ddiff.test.utils import *


# Torch
dtype = torch.float64
USE_CPU = True

# Constants
EPSILON = 10e-10

# Optimization Choices
OPTIMIZE_FH = False
OPTIMIZE_SH2 = False
OPTIMIZE_FH2 = False
OPTIMIZE_NHP = False
OPTIMIZE_THP = False
# Mesh/Program Input
OPTIMIZE_MESH = True
OPTIMIZE_VMESH = False #May be necessary, but unsure

# Factors for setting initial values for optimization. 
INIT_FACTOR = 1.2 # initial variables are multiplied by this factor as a starting point
OFFSET_FACTOR = 0.5 # initial variables are offset by themselves times this factor as a starting point
LOSS_FUNC = "sym" # "log" or "sym"

# Iteration Parameters
NUM_ITERS = 500
NUM_THREADS = 2
CLIP_NORM = 1e-0

# Learning Rate Parameters
INITIAL_LR = 1e-2
CYCLE_LR = True
LR_CYCLE_COUNT = 1
LR_CYCLE = math.ceil(NUM_ITERS // LR_CYCLE_COUNT)
MIN_LR = 1e-5

# Gif parameters
GENERATE_GIF = True
GIF_FPS = 5
GIF_FREQ = 25

# Folder Settings
now = datetime.now()
exp = np.floor(np.log10(np.abs(INITIAL_LR)))
base = int(INITIAL_LR*(10**(-exp)))
exp = int(exp)
folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")+"_"+str(NUM_ITERS)+f'_{base}e{exp}'+"_Cycle-"+str(CYCLE_LR)

local_dir = "kn1ddiff/test/h2_proc/"
run_dir = local_dir+"Runs/"+folder_name+"/"
image_dir = run_dir+"Images/"
in_file = "kh2_proc_in.json"
out_file = "kh2_proc_out.json"


if __name__ == "__main__":

    # Start logging to run file
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
    print("    FH: ", OPTIMIZE_FH)
    print("    SH2: ", OPTIMIZE_SH2)
    print("    FH2: ", OPTIMIZE_FH)
    print("    NHP: ", OPTIMIZE_NHP)
    print("    THP: ", OPTIMIZE_THP)
    print("    MESH: ", OPTIMIZE_MESH)
    print("    VMESH: ", OPTIMIZE_VMESH)
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
    with open(local_dir+in_file, 'r') as f:
        in_data = json.load(f)
        for key, value in in_data.items():
            in_data[key] = torch.tensor(value, dtype=dtype, device=device)
    with open(local_dir+out_file, 'r') as f:
        out_data = json.load(f)
        for key, value in out_data.items():
            out_data[key] = torch.tensor(value, dtype=dtype, device=device)

    # --- Load Mesh Outputs ---
    with open(local_dir+"h2_mesh_out.json", 'r') as f:
        mesh_output = json.load(f)
        for key, value in mesh_output.items():
            mesh_output[key] = torch.tensor(value, dtype=dtype, device=device)

    
    # Fixed

    # Gradient
    truein_fH = in_data["fH"]
    print_torch_range(truein_fH, "fH")
    truein_SH2 = in_data["SH2"]
    print_torch_range(truein_SH2, "SH2")
    truein_fH2 = in_data["fH2"]
    print_torch_range(truein_fH2, "fH2")
    truein_nHP = in_data["nHP"]
    print_torch_range(truein_nHP, "nHP")
    truein_THP = in_data["THP"]
    print_torch_range(truein_THP, "THP")

    truein_Ti = mesh_output["Ti"]
    print_torch_range(truein_Ti, "Ti")
    truein_Te = mesh_output["Te"]
    print_torch_range(truein_Te, "Te")
    truein_ne = mesh_output["ne"]
    print_torch_range(truein_ne, "ne")
    truein_Tnorm = mesh_output["Tnorm"]
    print("Tnorm: ", truein_Tnorm.item())

    truein_vx = mesh_output["vx"]
    print_torch_range(truein_vx, "vx")
    truein_vr = mesh_output["vr"]
    print_torch_range(truein_vr, "vr")
    print()
    # input()


    # Desired Outputs
    
    trueout_fH2 = out_data["fH2"]
    trueout_nHP = out_data["nHP"]
    trueout_THP = out_data["THP"]
    trueout_nH2 = out_data["nH2"]
    trueout_GammaxH2 = out_data["GammaxH2"]
    trueout_VxH2 = out_data["VxH2"]
    trueout_pH2 = out_data["pH2"]
    trueout_TH2 = out_data["TH2"]
    trueout_qxH2 = out_data["qxH2"]
    trueout_qxH2_total = out_data["qxH2_total"]
    trueout_Sloss = out_data["Sloss"]
    trueout_QH2 = out_data["QH2"]
    trueout_RxH2 = out_data["RxH2"]
    trueout_QH2_total = out_data["QH2_total"]
    trueout_AlbedoH2 = out_data["AlbedoH2"]
    trueout_WallH2 = out_data["WallH2"]
    trueout_fSH = out_data["fSH"]
    trueout_SH = out_data["SH"]
    trueout_SP = out_data["SP"]
    trueout_SHP = out_data["SHP"]
    trueout_NuE = out_data["NuE"]
    trueout_NuDis = out_data["NuDis"]
    trueout_ESH = out_data["ESH"]
    trueout_Eaxis = out_data["Eaxis"]


    # --- Set up Kinetic_H ---
    
    # --- Set up Kinetic_H ---

    with open(local_dir+"h2_mesh_in.json", 'r') as f:
        mesh_input = json.load(f)
        for key, value in mesh_input.items():
            mesh_input[key] = np.asarray(value)

    with open(local_dir+"kinetic_h2_in.json", 'r') as f:
        kh2_in = json.load(f)
        for key, value in kh2_in.items():
            kh2_in[key] = torch.tensor(value, dtype=dtype, device=device)
    
    mesh = KineticMesh('h2', mesh_input["mu"], mesh_input["x"], mesh_input["Ti"], mesh_input["Te"], mesh_input["n"], mesh_input["PipeDia"], E0=mesh_input["E0"], fctr=mesh_input["fctr"], device=device, dtype=dtype)
    
    kinetic_h2 = KineticH2(mesh, kh2_in["mu"], kh2_in["vxi"], kh2_in["fH2BC"], kh2_in["GammaxH2BC"], kh2_in["NuLoss"], kh2_in["SH2_initial"], 
                        compute_h_source=True, ni_correct=True, truncate=1.0e-3, max_gen=100, 
                        compute_errors=True, debrief=False, debug=False, 
                        device=device, dtype=dtype)


    # --- Test Input Data ---

    kh2_results = kinetic_h2.run_procedure(truein_fH, truein_SH2, truein_fH2, truein_nHP, truein_THP)
    check_close("fH2", kh2_results.fH2, trueout_fH2)
    check_close("nHP", kh2_results.nHP, trueout_nHP)
    check_close("THP", kh2_results.THP, trueout_THP)
    check_close("nH2", kh2_results.nH2, trueout_nH2)
    check_close("GammaxH2", kh2_results.GammaxH2, trueout_GammaxH2)
    check_close("VxH2", kh2_results.VxH2, trueout_VxH2)
    check_close("pH2", kh2_results.pH2, trueout_pH2)
    check_close("TH2", kh2_results.TH2, trueout_TH2)
    check_close("qxH2", kh2_results.qxH2, trueout_qxH2)
    check_close("qxH2_total", kh2_results.qxH2_total, trueout_qxH2_total)
    check_close("Sloss", kh2_results.Sloss, trueout_Sloss)
    check_close("QH2", kh2_results.QH2, trueout_QH2)
    check_close("RxH2", kh2_results.RxH2, trueout_RxH2)
    check_close("QH2_total", kh2_results.QH2_total, trueout_QH2_total)
    check_close("AlbedoH2", kh2_results.AlbedoH2, trueout_AlbedoH2)
    check_close("WallH2", kh2_results.WallH2, trueout_WallH2)
    check_close("fSH", kh2_results.fSH, trueout_fSH)
    check_close("SH", kh2_results.SH, trueout_SH)
    check_close("SP", kh2_results.SP, trueout_SP)
    check_close("SHP", kh2_results.SHP, trueout_SHP)
    check_close("NuE", kh2_results.NuE, trueout_NuE)
    check_close("NuDis", kh2_results.NuDis, trueout_NuDis)
    check_close("ESH", kh2_results.ESH, trueout_ESH)
    check_close("Eaxis", kh2_results.Eaxis, trueout_Eaxis)
    print()
    # input()

    trueout_results = kh2_results


    # --- Optimization Parameters ---
    
    parameterize = lambda tensor : torch.nn.Parameter(torch.log(torch.abs(tensor)))


    initial_fH = init_optimization_tensor(truein_fH, INIT_FACTOR, OFFSET_FACTOR)
    fH_param = parameterize(initial_fH)
    initial_SH2 = init_optimization_tensor(truein_SH2, INIT_FACTOR, OFFSET_FACTOR)
    SH2_param = parameterize(initial_SH2)
    initial_fH2 = init_optimization_tensor(truein_fH2, INIT_FACTOR, OFFSET_FACTOR)
    fH2_param = parameterize(initial_fH2)
    initial_nHP = init_optimization_tensor(truein_nHP, INIT_FACTOR, OFFSET_FACTOR)
    nHP_param = parameterize(initial_nHP)
    initial_THP = init_optimization_tensor(truein_THP, INIT_FACTOR, OFFSET_FACTOR)
    THP_param = parameterize(initial_THP)

    initial_Ti = init_optimization_tensor(truein_Ti, INIT_FACTOR, OFFSET_FACTOR)
    Ti_param = parameterize(initial_Ti)
    initial_Te = init_optimization_tensor(truein_Te, INIT_FACTOR, OFFSET_FACTOR)
    Te_param = parameterize(initial_Te)
    initial_ne = init_optimization_tensor(truein_ne, INIT_FACTOR, OFFSET_FACTOR)
    ne_param = parameterize(initial_ne)
    # initial_Tnorm = init_optimization_tensor(truein_Tnorm, INIT_FACTOR, OFFSET_FACTOR)
    # Tnorm_param = parameterize(initial_Tnorm)

    initial_vr = init_optimization_tensor(truein_vr, INIT_FACTOR, OFFSET_FACTOR)
    vr_param = parameterize(initial_vr)
    initial_vx = init_optimization_tensor(truein_vx, INIT_FACTOR, OFFSET_FACTOR)
    vx_param = parameterize(initial_vx)


    parameters = []

    if OPTIMIZE_FH:
        parameters.append(fH_param)
    if OPTIMIZE_SH2:
        parameters.extend([SH2_param])
    if OPTIMIZE_FH2:
        parameters.extend([fH2_param])
    if OPTIMIZE_NHP:
        parameters.extend([nHP_param])
    if OPTIMIZE_THP:
        parameters.extend([THP_param])
    
    if OPTIMIZE_MESH:
        parameters.extend([Ti_param, Te_param, ne_param])
        # parameters.extend([Ti_param, Te_param, ne_param, Tnorm_param])
    if OPTIMIZE_VMESH:
        parameters.extend([vr_param, vx_param])


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
        if OPTIMIZE_FH:
            fH_gifgen = GIF_Generator(NUM_ITERS, image_dir+"fH/", "fH", truein_fH[0,10,:], fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_SH2:
            SH2_gifgen = GIF_Generator(NUM_ITERS, image_dir+"SH2/", "SH2", truein_SH2[0,10,:], fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_FH2:
            fH2_gifgen = GIF_Generator(NUM_ITERS, image_dir+"fH2/", "fH2", truein_fH2[0,10,:], fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_NHP:
            nHP_gifgen = GIF_Generator(NUM_ITERS, image_dir+"nHP/", "nHP", truein_nHP, fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_THP:
            THP_gifgen = GIF_Generator(NUM_ITERS, image_dir+"THP/", "THP", truein_THP, fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_MESH:
            Ti_gifgen = GIF_Generator(NUM_ITERS, image_dir+"Ti/", "Ti", truein_Ti, fps=GIF_FPS, frequency=GIF_FREQ)
            Te_gifgen = GIF_Generator(NUM_ITERS, image_dir+"Te/", "Te", truein_Te, fps=GIF_FPS, frequency=GIF_FREQ)
            ne_gifgen = GIF_Generator(NUM_ITERS, image_dir+"ne/", "ne", truein_ne, fps=GIF_FPS, frequency=GIF_FREQ)
            # Tnorm_gifgen = GIF_Generator(NUM_ITERS, image_dir+"Tnorm/", "Tnorm", truein_Tnorm, fps=GIF_FPS, frequency=GIF_FREQ)


    # Capture Best Epoch
    loss_list = []
    lr_list = []
    best_loss = np.inf
    best_epoch = 0

    optim_start = time.time()

    for epoch in range(NUM_ITERS):

        epoch_start = time.time()

        # --- Bound Inputs ---

        fH_in = torch.sign(initial_fH)*torch.exp(fH_param) if OPTIMIZE_FH else truein_fH
        SH2_in = torch.sign(initial_SH2)*torch.exp(SH2_param) if OPTIMIZE_SH2 else truein_SH2
        fH2_in = torch.sign(initial_fH2)*torch.exp(fH2_param) if OPTIMIZE_FH2 else truein_fH2
        nHP_in = torch.exp(nHP_param) if OPTIMIZE_NHP else truein_nHP
        THP_in = torch.exp(THP_param) if OPTIMIZE_THP else truein_THP

        if OPTIMIZE_MESH:
            Ti_in = torch.exp(Ti_param)
            Te_in = torch.exp(Te_param)
            ne_in = torch.exp(ne_param)
            # Tnorm_in = torch.exp(Tnorm_param)

            mesh.Ti = Ti_in
            mesh.Te = Te_in
            mesh.ne = ne_in
            mesh.Tnorm = torch.nanmean(Ti_in) #Tnorm calculation in mesh

            # Reinitialize kinetic_h with mesh
            kinetic_h2 = KineticH2(mesh, kh2_in["mu"], kh2_in["vxi"], kh2_in["fH2BC"], kh2_in["GammaxH2BC"], kh2_in["NuLoss"], kh2_in["SH2_initial"], 
                        compute_h_source=True, ni_correct=True, truncate=1.0e-3, max_gen=100, 
                        compute_errors=True, debrief=False, debug=False, 
                        device=device, dtype=dtype)


        # --- Run Function ---
        kh_results = kinetic_h2.run_procedure(fH_in, SH2_in, fH2_in, nHP_in, THP_in)

        forward_done = time.time()
        forward_time = forward_done - epoch_start


        # --- Optimize ---

        # Compute Loss
        loss1 = loss_fun(kh_results.fH2, trueout_results.fH2)
        loss2 = loss_fun(kh_results.nHP, trueout_results.nHP)
        loss3 = loss_fun(kh_results.THP, trueout_results.THP)
        loss4 = loss_fun(kh_results.nH2, trueout_results.nH2)
        loss5 = loss_fun(kh_results.GammaxH2, trueout_results.GammaxH2)
        loss6 = loss_fun(kh_results.VxH2, trueout_results.VxH2)
        loss7 = loss_fun(kh_results.pH2, trueout_results.pH2)
        loss8 = loss_fun(kh_results.TH2, trueout_results.TH2)
        loss9 = loss_fun(kh_results.qxH2, trueout_results.qxH2)
        loss10 = loss_fun(kh_results.qxH2_total, trueout_results.qxH2_total)
        loss11 = loss_fun(kh_results.Sloss, trueout_results.Sloss)
        loss12 = loss_fun(kh_results.QH2, trueout_results.QH2)
        loss13 = loss_fun(kh_results.RxH2, trueout_results.RxH2)
        loss14 = loss_fun(kh_results.QH2_total, trueout_results.QH2_total)
        loss15 = loss_fun(kh_results.AlbedoH2, trueout_results.AlbedoH2)
        loss16 = loss_fun(kh_results.WallH2, trueout_results.WallH2)
        loss17 = loss_fun(kh_results.fSH, trueout_results.fSH)
        loss18 = loss_fun(kh_results.SH, trueout_results.SH)
        loss19 = loss_fun(kh_results.SP, trueout_results.SP)
        loss20 = loss_fun(kh_results.SHP, trueout_results.SHP)
        loss21 = loss_fun(kh_results.NuE, trueout_results.NuE)
        loss22 = loss_fun(kh_results.NuDis, trueout_results.NuDis)
        loss23 = loss_fun(kh_results.ESH, trueout_results.ESH)
        loss24 = loss_fun(kh_results.Eaxis, trueout_results.Eaxis)
        
        loss = (loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10+loss11+loss12+loss13
                +loss14+loss15+loss16+loss17+loss18+loss19+loss20+loss21+loss22+loss23+loss24)

        # Backprop
        optimizer.zero_grad()
        loss.backward()

        # Clip Gradient
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=CLIP_NORM)

        #Optimize
        optimizer.step()
        if CYCLE_LR:
            # scheduler.step(loss)
            scheduler.step()

        backward_done = time.time()
        backward_time = backward_done - forward_done

        # Save Best Epoch
        loss_list.append(loss.item())
        lr_list.append(scheduler.get_last_lr())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_inputs = {
                            "fH" : fH_in.detach().cpu(),
                            "SH2" : SH2_in.detach().cpu(),
                            "fH2" : fH2_in.detach().cpu(),
                            "nHP" : nHP_in.detach().cpu(),
                            "THP" : THP_in.detach().cpu(),

                            "Ti" : mesh.Ti.detach().cpu(),
                            "Te" : mesh.Te.detach().cpu(),
                            "ne" : mesh.ne.detach().cpu(),
                            "Tnorm" : mesh.Tnorm.detach().cpu(),
                            }
            
            best_pred = KH2Results(
                                kh_results.fH2,
                                kh_results.nHP,
                                kh_results.THP,
                                kh_results.nH2,
                                kh_results.GammaxH2,
                                kh_results.VxH2,
                                kh_results.pH2,
                                kh_results.TH2,
                                kh_results.qxH2,
                                kh_results.qxH2_total,
                                kh_results.Sloss,
                                kh_results.QH2,
                                kh_results.RxH2,
                                kh_results.QH2_total,
                                kh_results.AlbedoH2,
                                kh_results.WallH2,
                                kh_results.fSH,
                                kh_results.SH,
                                kh_results.SP,
                                kh_results.SHP,
                                kh_results.NuE,
                                kh_results.NuDis,
                                kh_results.ESH,
                                kh_results.Eaxis,
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
            if OPTIMIZE_FH:
                fH_gifgen.update(fH_in[0,10,:], epoch)
            if OPTIMIZE_SH2:
                SH2_gifgen.update(SH2_in[0,10,:], epoch)
            if OPTIMIZE_FH2:
                fH2_gifgen.update(fH2_in[0,10,:], epoch)
            if OPTIMIZE_NHP:
                nHP_gifgen.update(nHP_in, epoch)
            if OPTIMIZE_THP:
                THP_gifgen.update(THP_in, epoch)

            if OPTIMIZE_MESH:
                Ti_gifgen.update(Ti_in, epoch)
                Te_gifgen.update(Te_in, epoch)
                ne_gifgen.update(ne_in, epoch)

    optimization_runtime = time.time() - optim_start
    print(f"Total Optimization Time: {timedelta(seconds=round(optimization_runtime))}")






    # --- Analysis ---

    opt_inputs = best_inputs
    opt_results = best_pred

    # --- Analyze ---
    print("Best Epoch: ", best_epoch)

    print("### Inputs Analysis ###")

    # Optimized Inputs Analysis

    if OPTIMIZE_FH:
        analyze_difference("fH", loss_fun, opt_inputs["fH"], truein_fH)
        print()
    if OPTIMIZE_SH2:
        analyze_difference("SH2", loss_fun, opt_inputs["SH2"], truein_SH2)
        print()
    if OPTIMIZE_FH2:
        analyze_difference("fH2", loss_fun, opt_inputs["fH2"], truein_fH2)
        print()
    if OPTIMIZE_NHP:
        analyze_difference("nHP", loss_fun, opt_inputs["nHP"], truein_nHP)
        print()
    if OPTIMIZE_THP:
        analyze_difference("THP", loss_fun, opt_inputs["THP"], truein_THP)
        print()

    if OPTIMIZE_MESH:
        analyze_difference("Ti", loss_fun, opt_inputs["Ti"], truein_Ti)
        print()
        analyze_difference("Te", loss_fun, opt_inputs["Te"], truein_Te)
        print()
        analyze_difference("ne", loss_fun, opt_inputs["ne"], truein_ne)
        print()
        analyze_difference("Tnorm", loss_fun, opt_inputs["Tnorm"], truein_Tnorm)
        print("Tnorm Opt: ", opt_inputs["Tnorm"].item())
        print("Tnorm True: ", truein_Tnorm.item())
        print()

    # Outputs Analysis

    print("### Outputs Analysis ###")

    analyze_difference("fH2", loss_fun, kh_results.fH2, trueout_results.fH2)
    analyze_difference("nHP", loss_fun, kh_results.nHP, trueout_results.nHP)
    analyze_difference("THP", loss_fun, kh_results.THP, trueout_results.THP)
    analyze_difference("nH2", loss_fun, kh_results.nH2, trueout_results.nH2)
    analyze_difference("GammaxH2", loss_fun, kh_results.GammaxH2, trueout_results.GammaxH2)
    analyze_difference("VxH2", loss_fun, kh_results.VxH2, trueout_results.VxH2)
    analyze_difference("pH2", loss_fun, kh_results.pH2, trueout_results.pH2)
    analyze_difference("TH2", loss_fun, kh_results.TH2, trueout_results.TH2)
    analyze_difference("qxH2", loss_fun, kh_results.qxH2, trueout_results.qxH2)
    analyze_difference("qxH2_total", loss_fun, kh_results.qxH2_total, trueout_results.qxH2_total)
    analyze_difference("Sloss", loss_fun, kh_results.Sloss, trueout_results.Sloss)
    analyze_difference("QH2", loss_fun, kh_results.QH2, trueout_results.QH2)
    analyze_difference("RxH2", loss_fun, kh_results.RxH2, trueout_results.RxH2)
    analyze_difference("QH2_total", loss_fun, kh_results.QH2_total, trueout_results.QH2_total)
    analyze_difference("AlbedoH2", loss_fun, kh_results.AlbedoH2, trueout_results.AlbedoH2)
    analyze_difference("WallH2", loss_fun, kh_results.WallH2, trueout_results.WallH2)
    analyze_difference("fSH", loss_fun, kh_results.fSH, trueout_results.fSH)
    analyze_difference("SH", loss_fun, kh_results.SH, trueout_results.SH)
    analyze_difference("SP", loss_fun, kh_results.SP, trueout_results.SP)
    analyze_difference("SHP", loss_fun, kh_results.SHP, trueout_results.SHP)
    analyze_difference("NuE", loss_fun, kh_results.NuE, trueout_results.NuE)
    analyze_difference("NuDis", loss_fun, kh_results.NuDis, trueout_results.NuDis)
    analyze_difference("ESH", loss_fun, kh_results.ESH, trueout_results.ESH)
    analyze_difference("Eaxis", loss_fun, kh_results.Eaxis, trueout_results.Eaxis)

    # --- Plot Generation --- 

    print("Generating Images and Gifs")

    # Runtime Data
    generate_loss_plot(image_dir, "Loss", loss_list, xlabel="Epoch", ylabel="Symmetrical Loss")
    generate_lr_plot(image_dir, "LR", lr_list, xlabel="Epoch", ylabel="Learning Rate")
    

    if OPTIMIZE_FH:
        x = range(opt_inputs["fH"][0,10,:].numel())
        for i in range(len(opt_inputs["fH"][0,:,0])):
            generate_compare_plot(image_dir+"fH/", "fH-"+str(i), x, opt_inputs["fH"][0,i,:], x, truein_fH[0,i,:], init_x=x, init_y=initial_fH[0,i,:])
    if OPTIMIZE_SH2:
        x = range(opt_inputs["SH2"][0,10,:].numel())
        for i in range(len(opt_inputs["SH2"][0,:,0])):
            generate_compare_plot(image_dir+"SH2/", "SH2-"+str(i), x, opt_inputs["SH2"][0,i,:], x, truein_SH2[0,i,:], init_x=x, init_y=initial_SH2[0,i,:])
    if OPTIMIZE_FH2:
        x = range(opt_inputs["fH2"][0,10,:].numel())
        for i in range(len(opt_inputs["fH2"][0,:,0])):
            generate_compare_plot(image_dir+"fH2/", "fH2-"+str(i), x, opt_inputs["fH2"][0,i,:], x, truein_fH2[0,i,:], init_x=x, init_y=initial_fH2[0,i,:])
    if OPTIMIZE_NHP:
        x = range(opt_inputs["nHP"].numel())
        generate_compare_plot(image_dir+"nHP/", "nHP", x, opt_inputs["nHP"], x, truein_nHP, init_x=x, init_y=initial_nHP)
    if OPTIMIZE_THP:
        x = range(opt_inputs["THP"].numel())
        generate_compare_plot(image_dir+"THP/", "THP", x, opt_inputs["THP"], x, truein_THP, init_x=x, init_y=initial_THP)

    if OPTIMIZE_MESH:
        x = range(mesh_output["x"].numel()) # Not representative of scale, for viewing
        generate_compare_plot(image_dir+"Ti/", "Ti", x, opt_inputs["Ti"], x, truein_Ti, init_x=x, init_y=initial_Ti)
        generate_compare_plot(image_dir+"Te/", "Te", x, opt_inputs["Te"], x, truein_Te, init_x=x, init_y=initial_Te)
        generate_compare_plot(image_dir+"ne/", "ne", x, opt_inputs["ne"], x, truein_ne, init_x=x, init_y=initial_ne)

    # --- Gif Generation ---
    if GENERATE_GIF:
        if OPTIMIZE_FH:
            fH_gifgen.generate_gif()
        if OPTIMIZE_SH2:
            SH2_gifgen.generate_gif()
        if OPTIMIZE_FH2:
            fH2_gifgen.generate_gif()
        if OPTIMIZE_NHP:
            nHP_gifgen.generate_gif()
        if OPTIMIZE_THP:
            THP_gifgen.generate_gif()

        if OPTIMIZE_MESH:
            Ti_gifgen.generate_gif()
            Te_gifgen.generate_gif()
            ne_gifgen.generate_gif()