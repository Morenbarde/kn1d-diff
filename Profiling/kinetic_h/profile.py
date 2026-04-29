import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import json
import time
from datetime import timedelta, datetime
import math
from dataclasses import asdict

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.kinetic_h import *
from Optimization.utils import *


# Torch
dtype = torch.float64
USE_CPU = True

# Constants
EPSILON = 10e-10

# Mesh/Program Input
OPTIMIZE_MESH = True

# Factors for setting initial values for optimization. 
INIT_FACTOR = 0.9 # initial variables are multiplied by this factor as a starting point
OFFSET_FACTOR = 0.0 # initial variables are offset by themselves times this factor as a starting point
LOSS_FUNC = "sym" # "log" or "sym"

# Iteration Parameters
NUM_ITERS = 10
WARM_ITERS = 5
PROFILE_ITERS = 1
NUM_THREADS = 2
CLIP_NORM = 1e-0

# Learning Rate Parameters
INITIAL_LR = 1e-3
CYCLE_LR = False
LR_CYCLE_COUNT = 1
LR_CYCLE = math.ceil(NUM_ITERS // LR_CYCLE_COUNT)
MIN_LR = 1e-6

# Gif parameters
GENERATE_GIF = False
GIF_FPS = 5
GIF_FREQ = 10


# Folder Settings
now = datetime.now()
exp = np.floor(np.log10(np.abs(INITIAL_LR)))
base = int(INITIAL_LR*(10**(-exp)))
exp = int(exp)
folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")+"_"+str(NUM_ITERS)+f'_{base}e{exp}'+"_Cycle-"+str(CYCLE_LR)

local_dir = "Profiling/kinetic_h/"
run_dir = local_dir+"Runs/"+folder_name+"/"
image_dir = run_dir+"Images/"
data_dir = run_dir+"Data/"
in_file = "kh_proc_in.json"
out_file = "kh_proc_out.json"


if __name__ == "__main__":

    # Start logging to run file
    setup_log(run_dir)

    print("Process PID: ", os.getpid())


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and not USE_CPU else "cpu")
    print("device: ", device)

    activities = [ProfilerActivity.CPU]
    if use_cuda and not USE_CPU:
        activities.append(ProfilerActivity.CUDA)
    # if use_cuda:
    #     torch.cuda.manual_seed(72)


    ### Print Run Info ###

    print()
    print("Optimizing: ")
    print("    MESH: ", OPTIMIZE_MESH)
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
    with open(local_dir+"h_mesh_out.json", 'r') as f:
        mesh_output = json.load(f)
        for key, value in mesh_output.items():
            mesh_output[key] = torch.tensor(value, dtype=dtype, device=device)

    
    # Fixed

    # Gradient
    truein_fH2 = in_data["fH2"]
    print_torch_range(truein_fH2, "fH2")
    truein_fSH = in_data["fSH"]
    print_torch_range(truein_fSH, "fSH")
    truein_fH = in_data["fH"]
    print_torch_range(truein_fH, "fH")
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
    
    trueout_fH = out_data["fH"]
    trueout_nH = out_data["nH"]
    trueout_GammaxH = out_data["GammaxH"]
    trueout_VxH = out_data["VxH"]
    trueout_pH = out_data["pH"]
    trueout_TH = out_data["TH"]
    trueout_qxH = out_data["qxH"]
    trueout_qxH_total = out_data["qxH_total"]
    trueout_NetHSource = out_data["NetHSource"]
    trueout_Sion = out_data["Sion"]
    trueout_QH = out_data["QH"]
    trueout_RxH = out_data["RxH"]
    trueout_QH_total = out_data["QH_total"]
    trueout_AlbedoH = out_data["AlbedoH"]
    trueout_SideWallH = out_data["SideWallH"]


    # --- Set up Kinetic_H ---
    
    # --- Set up Kinetic_H ---

    with open(local_dir+"h_mesh_in.json", 'r') as f:
        mesh_input = json.load(f)
        for key, value in mesh_input.items():
            mesh_input[key] = torch.tensor(value, dtype=dtype, device=device)

    with open(local_dir+"kinetic_h_in.json", 'r') as f:
        kh_in = json.load(f)
        for key, value in kh_in.items():
            kh_in[key] = torch.tensor(value, dtype=dtype, device=device)
    
    mesh = KineticMesh('h', mesh_input["mu"], mesh_input["x"], mesh_input["Ti"], mesh_input["Te"], mesh_input["n"], mesh_input["PipeDia"], E0=mesh_input["E0"], fctr=mesh_input["fctr"])
    
    kinetic_h = KineticH(mesh, kh_in["mu"], kh_in["vxi"], kh_in["fHBC"], kh_in["GammaxHBC"], 
                        ni_correct=True, truncate=1.0e-3, max_gen=100, 
                        compute_errors=True, debrief=False, debug=False, 
                        device=device, dtype=dtype)


    # --- Test Input Data ---

    kh_results = kinetic_h.run_procedure(truein_fH2, truein_fSH, truein_fH, truein_nHP, truein_THP)
    check_close("fH", kh_results.fH, trueout_fH)
    check_close("nH", kh_results.nH, trueout_nH)
    check_close("GammaxH", kh_results.GammaxH, trueout_GammaxH)
    check_close("VxH", kh_results.VxH, trueout_VxH)
    check_close("pH", kh_results.pH, trueout_pH)
    check_close("TH", kh_results.TH, trueout_TH)
    check_close("qxH", kh_results.qxH, trueout_qxH)
    check_close("qxH_total", kh_results.qxH_total, trueout_qxH_total)
    check_close("NetHSource", kh_results.NetHSource, trueout_NetHSource)
    check_close("Sion", kh_results.Sion, trueout_Sion)
    check_close("QH", kh_results.QH, trueout_QH)
    check_close("RxH", kh_results.RxH, trueout_RxH)
    check_close("QH_total", kh_results.QH_total, trueout_QH_total)
    check_close("AlbedoH", kh_results.AlbedoH, trueout_AlbedoH)
    check_close("SideWallH", kh_results.SideWallH, trueout_SideWallH)
    print()
    # input()

    trueout_results = kh_results


    # --- Optimization Parameters ---
    
    parameterize = lambda tensor : torch.nn.Parameter(torch.log(torch.abs(tensor)))

    initial_Ti = init_optimization_tensor(truein_Ti, INIT_FACTOR, OFFSET_FACTOR)
    Ti_param = parameterize(initial_Ti)
    initial_Te = init_optimization_tensor(truein_Te, INIT_FACTOR, OFFSET_FACTOR)
    Te_param = parameterize(initial_Te)
    initial_ne = init_optimization_tensor(truein_ne, INIT_FACTOR, OFFSET_FACTOR)
    ne_param = parameterize(initial_ne)
    # initial_Tnorm = init_optimization_tensor(truein_Tnorm, INIT_FACTOR, OFFSET_FACTOR)
    # Tnorm_param = parameterize(initial_Tnorm)


    parameters = []
    if OPTIMIZE_MESH:
        parameters.extend([Ti_param, Te_param, ne_param])
        # parameters.extend([Ti_param, Te_param, ne_param, Tnorm_param])


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



    # --- Optimization Warmup ---

    fH2_in = truein_fH2
    fSH_in = truein_fSH
    fH_in = truein_fH
    nHP_in = truein_nHP
    THP_in = truein_THP

    optim_start = time.time()


    print("Warmup")

    for epoch in range(WARM_ITERS):

        epoch_start = time.time()

        # --- Bound Inputs ---

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
            kinetic_h = KineticH(mesh, kh_in["mu"], kh_in["vxi"], kh_in["fHBC"], kh_in["GammaxHBC"], 
                        ni_correct=True, truncate=1.0e-3, max_gen=100, 
                        compute_errors=True, debrief=False, debug=False, 
                        device=device, dtype=dtype)

        # --- Run Function ---
        kh_results = kinetic_h.run_procedure(fH2_in, fSH_in, fH_in, nHP_in, THP_in)

        forward_done = time.time()
        forward_time = forward_done - epoch_start


        # --- Optimize ---

        # Compute Loss
        loss1 = loss_fun(kh_results.fH, trueout_results.fH)
        loss2 = loss_fun(kh_results.nH, trueout_results.nH)
        loss3 = loss_fun(kh_results.GammaxH, trueout_results.GammaxH)
        loss4 = loss_fun(kh_results.VxH, trueout_results.VxH)
        loss5 = loss_fun(kh_results.pH, trueout_results.pH)
        loss6 = loss_fun(kh_results.TH, trueout_results.TH)
        loss7 = loss_fun(kh_results.qxH, trueout_results.qxH)
        loss8 = loss_fun(kh_results.qxH_total, trueout_results.qxH_total)
        loss9 = loss_fun(kh_results.NetHSource, trueout_results.NetHSource)
        loss10 = loss_fun(kh_results.Sion, trueout_results.Sion)
        loss11 = loss_fun(kh_results.QH, trueout_results.QH)
        loss12 = loss_fun(kh_results.RxH, trueout_results.RxH)
        loss13 = loss_fun(kh_results.QH_total, trueout_results.QH_total)
        loss14 = loss_fun(kh_results.AlbedoH, trueout_results.AlbedoH)
        loss15 = loss_fun(kh_results.SideWallH, trueout_results.SideWallH)
        
        loss = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10+loss11+loss12+loss13+loss14+loss15

            
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


        epoch_runtime = time.time() - epoch_start

        print(
            f"epoch: {epoch:<5} | "
            f"runtime: {epoch_runtime:<5.2f}   F: {forward_time:<5.2f}  B: {backward_time:<5.2f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | "
            f"loss: {loss.item():<10.6e}"
        )



    print("Profiling")

    for epoch in range(PROFILE_ITERS):

        epoch_start = time.time()

        # --- Bound Inputs ---

        with profile(
            activities=[ProfilerActivity.CPU],
            with_stack=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
            # record_shapes=False,
            # profile_memory=False,
        ) as prof:

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
                kinetic_h = KineticH(mesh, kh_in["mu"], kh_in["vxi"], kh_in["fHBC"], kh_in["GammaxHBC"], 
                            ni_correct=True, truncate=1.0e-3, max_gen=100, 
                            compute_errors=True, debrief=False, debug=False, 
                            device=device, dtype=dtype)

            # --- Run Function ---
            kh_results = kinetic_h.run_procedure(fH2_in, fSH_in, fH_in, nHP_in, THP_in)

        print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=20))
        
        forward_done = time.time()
        forward_time = forward_done - epoch_start


        # --- Optimize ---

        # Compute Loss
        loss1 = loss_fun(kh_results.fH, trueout_results.fH)
        loss2 = loss_fun(kh_results.nH, trueout_results.nH)
        loss3 = loss_fun(kh_results.GammaxH, trueout_results.GammaxH)
        loss4 = loss_fun(kh_results.VxH, trueout_results.VxH)
        loss5 = loss_fun(kh_results.pH, trueout_results.pH)
        loss6 = loss_fun(kh_results.TH, trueout_results.TH)
        loss7 = loss_fun(kh_results.qxH, trueout_results.qxH)
        loss8 = loss_fun(kh_results.qxH_total, trueout_results.qxH_total)
        loss9 = loss_fun(kh_results.NetHSource, trueout_results.NetHSource)
        loss10 = loss_fun(kh_results.Sion, trueout_results.Sion)
        loss11 = loss_fun(kh_results.QH, trueout_results.QH)
        loss12 = loss_fun(kh_results.RxH, trueout_results.RxH)
        loss13 = loss_fun(kh_results.QH_total, trueout_results.QH_total)
        loss14 = loss_fun(kh_results.AlbedoH, trueout_results.AlbedoH)
        loss15 = loss_fun(kh_results.SideWallH, trueout_results.SideWallH)
        
        loss = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10+loss11+loss12+loss13+loss14+loss15

        # Backprop
        with profile(
            activities=[ProfilerActivity.CPU],
            with_stack=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
            # record_shapes=False,
            # profile_memory=False,
        ) as prof:
            
            optimizer.zero_grad()
            loss.backward()

            # Clip Gradient
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=CLIP_NORM)

            #Optimize
            optimizer.step()
            if CYCLE_LR:
                # scheduler.step(loss)
                scheduler.step()

        print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=20))

        backward_done = time.time()
        backward_time = backward_done - forward_done


        epoch_runtime = time.time() - epoch_start

        print(
            f"epoch: {epoch:<5} | "
            f"runtime: {epoch_runtime:<5.2f}   F: {forward_time:<5.2f}  B: {backward_time:<5.2f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | "
            f"loss: {loss.item():<10.6e}"
        )
