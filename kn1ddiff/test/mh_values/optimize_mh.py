import torch
import numpy as np
import json

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.kinetic_h import KineticH
from kn1ddiff.test.utils import *



dir = "kn1ddiff/test/mh_values/"
data_file = "mh_in_out1.json"
generate_gif = False
num_iters = 50


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
    print("fH abs Range", torch.max(torch.abs(true_fH)), torch.min(torch.abs(true_fH)))
    print("nH Range: ", torch.max(true_nH), torch.min(true_nH))
    print("TH2_mom Range: ", torch.max(true_TH2_mom), torch.min(true_TH2_mom))
    print("VxH2_mom Range: ", torch.max(true_VxH2_mom), torch.min(true_VxH2_mom))
    input()

    # Desired Outputs
    
    true_mh_h = torch.asarray(data["MH_H"])
    true_mh_p = torch.asarray(data["MH_P"])
    true_mh_h2 = torch.asarray(data["MH_H2"])


    # Test input Data

    kinetic_h.H2_Moments.VxH2 = true_VxH2_mom
    kinetic_h.H2_Moments.TH2 = true_TH2_mom
    m_vals = kinetic_h._compute_mh_values(true_fH, true_nH)
    print(np.allclose(m_vals.H_H, true_mh_h), np.allclose(m_vals.H_P, true_mh_p), np.allclose(m_vals.H_H2, true_mh_h2))
    print(rel_L2_torch(m_vals.H_H, true_mh_h), rel_L2_torch(m_vals.H_P, true_mh_p), rel_L2_torch(m_vals.H_H2, true_mh_h2))
    input()


    # --- Test Optimization ---
    initial_fH = torch.nn.Parameter(torch.randn_like(true_fH))
    initial_nH = torch.nn.Parameter(torch.randn_like(true_nH))
    initial_TH2_mom = torch.nn.Parameter(torch.randn_like(true_TH2_mom))
    initial_VxH2_mom = torch.nn.Parameter(torch.randn_like(true_VxH2_mom))

    optimizer = torch.optim.Adam([initial_fH, initial_nH, initial_TH2_mom, initial_VxH2_mom], lr=1e-1)
    # optimizer = torch.optim.SGD([initial_fH, initial_nH, initial_TH2_mom, initial_VxH2_mom], lr=2e-1, momentum=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=30,
        min_lr=1e-6
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)

    loss_fun = lambda pred, true : torch.sqrt(torch.nn.functional.mse_loss(pred, true))

    # Init Gif Generator

    # Capture Best Epoch
    loss_list = []
    best_loss = np.inf
    best_epoch = 0
    kinetic_h.H2_Moments.VxH2 = true_VxH2_mom
    kinetic_h.H2_Moments.TH2 = true_TH2_mom
    for epoch in range(num_iters):
        fH = 1e19 * (2*torch.sigmoid(initial_fH) - 1)
        nH = 1e17 * torch.sigmoid(initial_nH)
        TH2_mom = 4 * torch.sigmoid(initial_TH2_mom)
        VxH2_mom = 15000 * torch.sigmoid(initial_VxH2_mom)

        kinetic_h.H2_Moments.VxH2 = VxH2_mom
        kinetic_h.H2_Moments.TH2 = TH2_mom
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
        # torch.nn.utils.clip_grad_norm_([initial_fH, initial_nH, initial_TH2_mom, initial_VxH2_mom], max_norm=1.0)

        #Optimize
        optimizer.step()
        scheduler.step(loss)

        # Save Best Epoch
        loss_list.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_inputs = [fH.detach().cpu(), nH.detach().cpu(), TH2_mom.detach().cpu(), VxH2_mom.detach().cpu()]
            best_pred = [m_vals.H_H.detach().cpu(), m_vals.H_P.detach().cpu(), m_vals.H_H2.detach().cpu()]
            best_epoch = epoch

        print(
            f"epoch: {epoch:<5} | "
            f"loss: {loss.item():<10.6e} | "
            f"learning rate: {optimizer.param_groups[0]['lr']:.2e}"
        )



    # --- Convert to numpy for analysis ---

    opt_fH, opt_nH, opt_TH2, opt_VxH2 = best_inputs[0], best_inputs[1], best_inputs[2], best_inputs[3]
    opt_MH_H, opt_MH_P, opt_MH_H2 = best_pred[0], best_pred[1], best_pred[2]

    fH_loss = loss_fun(opt_fH, true_fH).item()
    nH_loss = loss_fun(opt_nH, true_nH).item()
    TH2_loss = loss_fun(opt_TH2, true_TH2_mom).item()
    VxH2_loss = loss_fun(opt_VxH2, true_VxH2_mom).item()
    mh_h_loss = loss_fun(opt_MH_H, true_mh_h).item()
    mh_p_loss = loss_fun(opt_MH_P, true_mh_p).item()
    mh_h2_loss = loss_fun(opt_MH_H2, true_mh_h2).item()

    opt_fH, opt_nH, opt_TH2, opt_VxH2 = opt_fH.numpy(), opt_nH.numpy(), opt_TH2.numpy(), opt_VxH2.numpy()
    opt_MH_H, opt_MH_P, opt_MH_H2 = opt_MH_H.numpy(), opt_MH_P.numpy(), opt_MH_H2.numpy()

    true_fH, true_nH, true_TH2_mom, true_VxH2_mom = true_fH.numpy(), true_nH.numpy(), true_TH2_mom.numpy(), true_VxH2_mom.numpy()
    true_mh_h, true_mh_p, true_mh_h2 = true_mh_h.numpy(), true_mh_p.numpy(), true_mh_h2.numpy()

    # --- Analyze ---
    print("Best Epoch: ", best_epoch)

    print("fH Loss: ", fH_loss)
    print("fH Relative L2: ", rel_L2_np(opt_fH, true_fH))
    print()
    print("nH Loss: ", nH_loss)
    print("nH Relative L2: ", rel_L2_np(opt_nH, true_nH))
    print()
    print("TH2 Loss: ", TH2_loss)
    print("TH2 Relative L2: ", rel_L2_np(opt_TH2, true_TH2_mom))
    print()
    print("VxH2 Loss: ", VxH2_loss)
    print("VxH2 Relative L2: ", rel_L2_np(opt_VxH2, true_VxH2_mom))
    print()

    input()


    print("MH_H Loss: ", mh_h_loss)
    print("MH_H Relative L2: ", rel_L2_np(opt_MH_H, true_mh_h))
    print()
    print("MH_P Loss: ", mh_p_loss)
    print("MH_P Relative L2: ", rel_L2_np(opt_MH_P, true_mh_p))
    print()
    print("MH_H2 Loss: ", mh_h2_loss)
    print("MH_H2 Relative L2: ", rel_L2_np(opt_MH_H2, true_mh_h2))
    print()
    input()



    print("Generating Images and Gifs")

    x = range(opt_nH.size)
    generate_compare_plot(dir, "nH", x, opt_nH, x, true_nH)
    x = range(opt_TH2.size)
    generate_compare_plot(dir, "nH", x, opt_TH2, x, true_TH2_mom)
    x = range(opt_VxH2.size)
    generate_compare_plot(dir, "VxH2", x, opt_VxH2, x, true_VxH2_mom)

    generate_loss_plot(dir, "Loss", loss_list, xlabel="Epoch", ylabel="MSE Loss")

    # tmax_gifgen.generate_gif()
    # vx_shift_gifgen.generate_gif()