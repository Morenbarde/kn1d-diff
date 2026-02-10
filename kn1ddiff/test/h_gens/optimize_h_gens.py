import torch
import numpy as np
import json

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.kinetic_h import *
from kn1ddiff.test.utils import *



dir = "kn1ddiff/test/h_gens/"
in_file = "kh_gens_in.json"
out_file = "kh_gens_out.json"

# Constants
EPSILON = 10e-10

# Optimization Choices
OPTIMIZE_FH = True
OPTIMIZE_NH = True
OPTIMIZE_FHG = True
OPTIMIZE_NHG = True
OPTIMIZE_MESH_COEF = True
OPTIMIZE_COLLISION = True

# Learning Rate Parameters
INITIAL_LR = 1e-1
LR_CYCLE = 50
MIN_LR = 1e-6

# Iteration Parameters
NUM_ITERS = 200
CLIP_NORM = 1e-1

# Gif parameters
GENERATE_GIF = True
GIF_FPS = 10
GIF_FREQ = 10


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)
    # if use_cuda:
    #     torch.cuda.manual_seed(72)

    torch.set_default_dtype(torch.float64)


    # --- Load Inputs and Outputs ---
    with open(dir+in_file, 'r') as f:
        in_data = json.load(f)
    with open(dir+out_file, 'r') as f:
        out_data = json.load(f)

    mesh_input = np.load(dir+"h_mesh_in.npz")
    kh_in = np.load(dir+"kinetic_h_in.npz")

    # Fixed

    # Gradient
    truein_fH = torch.asarray(in_data["fH"])
    print("fH Range: ", torch.max(truein_fH), torch.min(truein_fH))
    truein_nH = torch.asarray(in_data["nH"])
    print("nH Range: ", torch.max(truein_nH), torch.min(truein_nH))
    truein_fHG = torch.asarray(in_data["fHG"])
    print("fHG Range: ", torch.max(truein_fHG), torch.min(truein_fHG))
    truein_NHG = torch.asarray(in_data["NHG"])
    print("NHG Range: ", torch.max(truein_NHG), torch.min(truein_NHG))
    truein_A = torch.asarray(in_data["A"])
    print("A Range: ", torch.max(truein_A), torch.min(truein_A))
    truein_B = torch.asarray(in_data["B"])
    print("B Range: ", torch.max(truein_B), torch.min(truein_B))
    truein_C = torch.asarray(in_data["C"])
    print("C Range: ", torch.max(truein_C), torch.min(truein_C))
    truein_D = torch.asarray(in_data["D"])
    print("D Range: ", torch.max(truein_D), torch.min(truein_D))
    truein_CF_H_H = torch.asarray(in_data["CF_H_H"])
    print("CF_H_H Range: ", torch.max(truein_CF_H_H), torch.min(truein_CF_H_H))
    truein_CF_H_P = torch.asarray(in_data["CF_H_P"])
    print("CF_H_P Range: ", torch.max(truein_CF_H_P), torch.min(truein_CF_H_P))
    truein_CF_H_H2 = torch.asarray(in_data["CF_H_H2"])
    print("CF_H_H2 Range: ", torch.max(truein_CF_H_H2), torch.min(truein_CF_H_H2))
    # input()
    truein_meq_coeffs = MeshEqCoefficients(truein_A, truein_B, truein_C, truein_D, 0, 0)
    truein_collision_freqs = CollisionType(truein_CF_H_H, truein_CF_H_P, truein_CF_H_H2)


    # Desired Outputs
    
    trueout_fH = torch.asarray(out_data["fH"])
    trueout_nH = torch.asarray(out_data["nH"])
    trueout_fHG = torch.asarray(out_data["fHG"])
    trueout_NHG = torch.asarray(out_data["NHG"])
    trueout_Beta_CX_sum = torch.asarray(out_data["Beta_CX_sum"])
    trueout_Msum_H_H = torch.asarray(out_data["Msum_H_H"])
    trueout_Msum_H_P = torch.asarray(out_data["Msum_H_P"])
    trueout_Msum_H_H2 = torch.asarray(out_data["Msum_H_H2"])

    x = range(len(truein_NHG[:,0]))
    generate_compare_plot(dir, "NHG_in", x, truein_NHG[:,0], x, trueout_NHG[:,0])


    # --- Set up Kinetic_H ---
    
    h_mesh = KineticMesh('h', mesh_input["mu"], mesh_input["x"], mesh_input["Ti"], mesh_input["Te"], mesh_input["n"], mesh_input["PipeDia"], E0=mesh_input["E0"], fctr=mesh_input["fctr"], param_type='torch')
    kinetic_h = KineticH(h_mesh, torch.from_numpy(kh_in["mu"]), torch.from_numpy(kh_in["vxiA"]), torch.from_numpy(kh_in["fHBC"]), torch.from_numpy(kh_in["GammaxHBC"]), 
                        ni_correct=True, truncate=1.0e-3, max_gen=100, 
                        compute_errors=True, debrief=True, debug=False)

    # kinetic_h internal Data
    kinetic_h.H2_Moments.TH2 = torch.asarray(in_data['TH2_Moment'])
    kinetic_h.H2_Moments.VxH2 = torch.asarray(in_data['VxH2_Moment'])
    kinetic_h.Internal.fi_hat = torch.asarray(in_data['fi_hat'])
    kinetic_h.Internal.Alpha_CX = torch.asarray(in_data['Alpha_CX'])
    kinetic_h.Internal.ni = torch.asarray(in_data['ni'])
    kinetic_h.Internal.SIG_CX = torch.asarray(in_data['SIG_CX'])


    # --- Test Input Data

    fH, nH, fHG, NHG, Beta_CX_sum, m_sums, igen = kinetic_h._run_generations(truein_fH, truein_nH, truein_fHG, truein_NHG, truein_meq_coeffs, truein_collision_freqs, True)
    print("fH close: ", np.allclose(fH, trueout_fH))
    print(rel_L2_torch(fH, trueout_fH))
    print("nH close: ", np.allclose(nH, trueout_nH))
    print(rel_L2_torch(nH, trueout_nH))
    print("fHG close: ", np.allclose(fHG, trueout_fHG))
    print(rel_L2_torch(fHG, trueout_fHG))
    print("NHG close: ", np.allclose(NHG, trueout_NHG))
    print(rel_L2_torch(NHG, trueout_NHG))
    for i in range(len(NHG[0,:])):
        x = range(len(NHG[:,0]))
        generate_compare_plot(dir, "NHG"+str(i), x, NHG[:,i], x, trueout_NHG[:,i])
    print("Beta_CX_sum close: ", np.allclose(Beta_CX_sum, trueout_Beta_CX_sum))
    print(rel_L2_torch(Beta_CX_sum, trueout_Beta_CX_sum))
    print("Msum_H_H close: ", np.allclose(m_sums.H_H, trueout_Msum_H_H))
    print(rel_L2_torch(m_sums.H_H, trueout_Msum_H_H))
    print("Msum_H_P close: ", np.allclose(m_sums.H_P, trueout_Msum_H_P))
    print(rel_L2_torch(m_sums.H_P, trueout_Msum_H_P))
    print("Msum_H_H2 close: ", np.allclose(m_sums.H_H2, trueout_Msum_H_H2))
    print(rel_L2_torch(m_sums.H_H2, trueout_Msum_H_H2))
    input()


    # --- Test Optimization ---
    initial_fH = torch.nn.Parameter(torch.randn_like(true_fH, requires_grad=True))
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
    # loss_fun = lambda pred, true : ((torch.log(pred + EPSILON) - torch.log(true + EPSILON))**2).mean()

    def symmetric_log(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    loss_fun = lambda pred, true : ((symmetric_log(pred) - symmetric_log(true))**2).mean()










    # --- Optimization ---

    # Init Gif Generator
    fh_gifgen = GIF_Generator(NUM_ITERS, dir+"fH_Images/", "fH", true_fH[0,10,:], fps=GIF_FPS, frequency=GIF_FREQ)

    # Capture Best Epoch
    loss_list = []
    lr_list = []
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(NUM_ITERS):

        if OPTIMIZE_FH:
            # fH = 1e19 * torch.nn.functional.softplus(initial_fH)
            # fH = 1e19*torch.nn.functional.tanh(initial_fH)
            fH = 1e19 * torch.sigmoid(initial_fH)
            # fH = initial_fH
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
            best_inputs = [fH.detach().cpu()]
            best_pred = [beta_cx.detach().cpu()]
            best_epoch = epoch

        print(
            f"epoch: {epoch:<5} | "
            f"loss: {loss.item():<10.6e} | "
            # f"learning rate: {optimizer.param_groups[0]['lr']:.2e}"
            f"learning rate: {scheduler.get_last_lr()[0]:.2e}"
        )

        # print("FH_2", fH[0,10,:])
        if GENERATE_GIF:
            fh_gifgen.update(fH[0,10,:], epoch)









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
        generate_compare_plot(dir, "fH"+str(i), x, opt_fH[0,i,:], x, true_fH[0,i,:])

    generate_loss_plot(dir, "Loss", loss_list, xlabel="Epoch", ylabel="Symmetrical Loss")
    generate_lr_plot(dir, "LR", lr_list, xlabel="Epoch", ylabel="Learning Rate")

    if GENERATE_GIF:
        if OPTIMIZE_FH:
            fh_gifgen.generate_gif()