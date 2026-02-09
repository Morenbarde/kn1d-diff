import torch
import numpy as np
import json

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.kinetic_h import KineticH
from kn1ddiff.test.utils import *



dir = "kn1ddiff/test/h_gens/"
data_file = "h_gens_in_out.json"
num_iters = 200
epsilon = 10e-10

OPTIMIZE_FH = False
OPTIMIZE_NH = True

# Learning Rate Parameters
INITIAL_LR = 2e-1
LR_CYCLE = 100
MIN_LR = 1e-6

# Iteration Parameters
CLIP_NORM = 1e0

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


    # --- Set up Kinetic_H ---
    mesh_input = np.load(dir+"h_mesh_in.npz")
    kh_in = np.load(dir+"kinetic_h_in.npz")

    h_mesh = KineticMesh('h', mesh_input["mu"], mesh_input["x"], mesh_input["Ti"], mesh_input["Te"], mesh_input["n"], mesh_input["PipeDia"], E0=mesh_input["E0"], fctr=mesh_input["fctr"], param_type='torch')
    kinetic_h = KineticH(h_mesh, torch.from_numpy(kh_in["mu"]), torch.from_numpy(kh_in["vxiA"]), torch.from_numpy(kh_in["fHBC"]), torch.from_numpy(kh_in["GammaxHBC"]), 
                        ni_correct=True, truncate=1.0e-3, max_gen=100, 
                        compute_errors=True, debrief=True, debug=False)


    # --- Load Inputs and Outputs ---
    with open(dir+data_file, 'r') as f:
        data = json.load(f)

    # Fixed

    # Gradient
    true_fH = torch.asarray(data["fH"])
    true_nH = torch.asarray(data["nH"])
    true_TH2_mom = torch.asarray(data["TH2_Moment"])
    true_VxH2_mom = torch.asarray(data["VxH2_Moment"])
    print("fH Range: ", torch.max(true_fH), torch.min(true_fH))
    # print("fH abs Range", torch.max(torch.abs(true_fH)), torch.min(torch.abs(true_fH)))
    print("nH Range: ", torch.max(true_nH), torch.min(true_nH))
    # print("TH2_mom Range: ", torch.max(true_TH2_mom), torch.min(true_TH2_mom))
    # print("VxH2_mom Range: ", torch.max(true_VxH2_mom), torch.min(true_VxH2_mom))
    # input()

    # Desired Outputs
    
    true_mh_h = torch.asarray(data["MH_H"])
    true_mh_p = torch.asarray(data["MH_P"])
    true_mh_h2 = torch.asarray(data["MH_H2"])


    # Test input Data

    kinetic_h.H2_Moments.VxH2 = true_VxH2_mom
    kinetic_h.H2_Moments.TH2 = true_TH2_mom
    m_vals = kinetic_h._compute_mh_values(true_fH, true_nH)
    # output1, output2 = kinetic_h._compute_mh_values(true_fH, true_nH)
    # print(np.allclose(m_vals.H_H, true_mh_h), np.allclose(m_vals.H_P, true_mh_p), np.allclose(m_vals.H_H2, true_mh_h2))
    # print(rel_L2_torch(m_vals.H_H, true_mh_h), rel_L2_torch(m_vals.H_P, true_mh_p), rel_L2_torch(m_vals.H_H2, true_mh_h2))
    # input()


    # --- Test Optimization ---
    # initial_fH = torch.nn.Parameter(torch.randn_like(true_fH, requires_grad=True))
    initial_fH = torch.nn.Parameter(torch.zeros_like(true_fH, requires_grad=True))
    initial_nH = torch.nn.Parameter(torch.randn_like(true_nH, requires_grad=True))
    # initial_nH = torch.nn.Parameter(torch.zeros_like(true_nH, requires_grad=True))
    initial_TH2_mom = torch.nn.Parameter(torch.randn_like(true_TH2_mom))
    initial_VxH2_mom = torch.nn.Parameter(torch.randn_like(true_VxH2_mom))


    parameters = []
    if OPTIMIZE_FH:
        parameters.append(initial_fH)
    if OPTIMIZE_NH:
        parameters.append(initial_nH)

    optimizer = torch.optim.Adam(parameters, lr=INITIAL_LR, betas=(0.9,  0.99))

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     factor=0.1,
    #     patience=50,
    #     min_lr=1e-5
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=LR_CYCLE,
        # T_mult=1,
        # eta_min=MIN_LR,
    )


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



    # Init Gif Generator
    fh_gifgen = GIF_Generator(num_iters, dir+"fH_Images/", "fH", true_fH[0,0,:], fps=GIF_FPS, frequency=GIF_FREQ)
    nh_gifgen = GIF_Generator(num_iters, dir+"nH_Images/", "nH", true_nH, fps=GIF_FPS, frequency=GIF_FREQ)

    # Capture Best Epoch
    loss_list = []
    lr_list = []
    best_loss = np.inf
    best_epoch = 0
    kinetic_h.H2_Moments.VxH2 = true_VxH2_mom
    kinetic_h.H2_Moments.TH2 = true_TH2_mom
    for epoch in range(num_iters):

        if OPTIMIZE_FH:
            # fH = 1e19 * torch.nn.functional.softplus(initial_fH)
            fH = 1e19*torch.nn.functional.tanh(initial_fH)
            # fH = 1e19 * torch.sigmoid(initial_fH)
        else:
            fH = true_fH

        if OPTIMIZE_NH:
            nH = 1e17 * torch.sigmoid(initial_nH)
        else:
            nH = true_nH

    
        m_vals = kinetic_h._compute_mh_values(fH, nH)

        # Compute Loss
        loss1 = loss_fun(m_vals.H_H, true_mh_h)
        loss2 = loss_fun(m_vals.H_P, true_mh_p)
        loss3 = loss_fun(m_vals.H_H2, true_mh_h2)
        loss = loss1 + loss2 + loss3

        # Backprop
        optimizer.zero_grad()
        loss.backward()

        # Clip Gradient
        torch.nn.utils.clip_grad_norm_([initial_fH, initial_nH], max_norm=CLIP_NORM)

        #Optimize
        optimizer.step()
        # scheduler.step(loss)
        scheduler.step()

        # Save Best Epoch
        loss_list.append(loss.item())
        lr_list.append(scheduler.get_last_lr())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_inputs = [fH.detach().cpu(), nH.detach().cpu()]
            best_pred = [m_vals.H_H.detach().cpu(), m_vals.H_P.detach().cpu(), m_vals.H_H2.detach().cpu()]
            best_epoch = epoch

        print(
            f"epoch: {epoch:<5} | "
            f"loss: {loss.item():<10.6e} | "
            # f"learning rate: {optimizer.param_groups[0]['lr']:.2e}"
            f"learning rate: {scheduler.get_last_lr()[0]:.2e}"
        )

        # print("FH_2", fH[:,0,0])
        if GENERATE_GIF:
            fh_gifgen.update(fH[0,0,:], epoch)
            nh_gifgen.update(nH, epoch)



    # --- Convert to numpy for analysis ---

    opt_fH, opt_nH, opt_TH2, opt_VxH2 = best_inputs[0], best_inputs[1], true_TH2_mom, true_VxH2_mom
    opt_MH_H, opt_MH_P, opt_MH_H2 = best_pred[0], best_pred[1], best_pred[2]

    fH_loss = loss_fun(opt_fH, true_fH).item()
    nH_loss = loss_fun(opt_nH, true_nH).item()
    # TH2_loss = loss_fun(opt_TH2, true_TH2_mom).item()
    # VxH2_loss = loss_fun(opt_VxH2, true_VxH2_mom).item()
    mh_h_loss = loss_fun(opt_MH_H, true_mh_h).item()
    mh_p_loss = loss_fun(opt_MH_P, true_mh_p).item()
    mh_h2_loss = loss_fun(opt_MH_H2, true_mh_h2).item()

    # --- Analyze ---
    print("Best Epoch: ", best_epoch)

    if OPTIMIZE_FH:
        print("fH Loss: ", fH_loss)
        print("fH Relative L2: ", rel_L2_torch(opt_fH, true_fH))
        print()
    if OPTIMIZE_NH:
        print("nH Loss: ", nH_loss)
        print("nH Relative L2: ", rel_L2_torch(opt_nH, true_nH))
        print()
    # print("TH2 Loss: ", TH2_loss)
    # print("TH2 Relative L2: ", rel_L2_torch(opt_TH2, true_TH2_mom))
    # print()
    # print("VxH2 Loss: ", VxH2_loss)
    # print("VxH2 Relative L2: ", rel_L2_torch(opt_VxH2, true_VxH2_mom))
    # print()

    # input()


    print("MH_H Loss: ", mh_h_loss)
    print("MH_H Relative L2: ", rel_L2_torch(opt_MH_H, true_mh_h))
    print()
    print("MH_P Loss: ", mh_p_loss)
    print("MH_P Relative L2: ", rel_L2_torch(opt_MH_P, true_mh_p))
    print()
    print("MH_H2 Loss: ", mh_h2_loss)
    print("MH_H2 Relative L2: ", rel_L2_torch(opt_MH_H2, true_mh_h2))
    print()
    # input()



    print("Generating Images and Gifs")

    x = range(opt_fH[0,0,:].numel())
    generate_compare_plot(dir, "fH", x, opt_fH[0,0,:], x, true_fH[0,0,:])
    x = range(opt_nH.numel())
    generate_compare_plot(dir, "nH", x, opt_nH, x, true_nH)
    # x = range(opt_TH2.size)
    # generate_compare_plot(dir, "TH2", x, opt_TH2, x, true_TH2_mom)
    # x = range(opt_VxH2.size)
    # generate_compare_plot(dir, "VxH2", x, opt_VxH2, x, true_VxH2_mom)

    generate_loss_plot(dir, "Loss", loss_list, xlabel="Epoch", ylabel="Symmetrical Loss")
    generate_lr_plot(dir, "LR", lr_list, xlabel="Epoch", ylabel="Learning Rate")

    if GENERATE_GIF:
        if OPTIMIZE_FH:
            fh_gifgen.generate_gif()
        if OPTIMIZE_NH:
            nh_gifgen.generate_gif()