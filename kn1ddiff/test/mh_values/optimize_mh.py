import torch
import numpy as np
import json

from kn1ddiff.kinetic_mesh import *
from kn1ddiff.kinetic_h import KineticH
from kn1ddiff.test.utils import *



dir = "kn1ddiff/test/mh_values/"
data_file = "mh_in_out1.json"
generate_gif = False
num_iters = 500


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
    print("nH Range: ", torch.max(true_nH), torch.min(true_nH))
    print("TH2_mom Range: ", torch.max(true_TH2_mom), torch.min(true_TH2_mom))
    print("VxH2_mom Range: ", torch.max(true_VxH2_mom), torch.min(true_VxH2_mom))
    input()

    # Desired Outputs
    
    true_mh_h = torch.asarray(data["MH_H"])
    true_mh_p = torch.asarray(data["MH_P"])
    true_mh_h2 = torch.asarray(data["MH_H2"])


    # --- Test Optimization ---
    initial_fH = torch.nn.Parameter(torch.randn_like(true_fH))
    initial_nH = torch.nn.Parameter(torch.randn_like(true_nH))
    initial_TH2_mom = torch.nn.Parameter(torch.randn_like(true_TH2_mom))
    initial_VxH2_mom = torch.nn.Parameter(torch.randn_like(true_VxH2_mom))

    optimizer = torch.optim.Adam([initial_fH, initial_nH, initial_TH2_mom, initial_VxH2_mom], lr=2e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=30,
        min_lr=1e-6
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)

    loss_fun = torch.nn.MSELoss()

    # Init Gif Generator

    # Capture Best Epoch
    loss_list = []
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(num_iters):
        fH = 10e19 * (2*torch.sigmoid(initial_fH) - 1)
        nH = 10e17 * torch.sigmoid(initial_nH)
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
        torch.nn.utils.clip_grad_norm_([initial_fH, initial_nH, initial_TH2_mom, initial_VxH2_mom], max_norm=1.0)

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

    tmax, shift = best_inputs[0], best_inputs[1]
    maxwell = best_pred

    tmax_loss = torch.nn.functional.mse_loss(tmax, Tmaxwell).item()
    vxshift_loss = torch.nn.functional.mse_loss(shift, vx_shift).item()
    max_loss = torch.nn.functional.mse_loss(maxwell, maxwell_old).item()

    true_tmax = Tmaxwell.cpu().detach().numpy()
    opt_tmax = tmax.cpu().detach().numpy()

    true_vxshift = vx_shift.cpu().detach().numpy()
    opt_vxshift = shift.cpu().detach().numpy()

    true_maxwell = maxwell_old.cpu().detach().numpy()
    opt_maxwell = maxwell.cpu().detach().numpy()


    # --- Analyze ---
    print("Best Epoch: ", best_epoch)

    print("True Tmaxwell Range")
    print(np.min(true_tmax), np.max(true_tmax))
    print("Calculated Tmaxwell Range")
    print(np.min(opt_tmax), np.max(opt_tmax))
    print()
    print("True vx_shift range")
    print(np.min(true_vxshift), np.max(true_vxshift))
    print("Calculated vx_shift range")
    print(np.min(opt_vxshift), np.max(opt_vxshift))
    input()

    print("True Tmaxwell")
    print(true_tmax)
    print("Calculated Tmaxwell")
    print(opt_tmax)
    print("Tmaxwell Loss: ", tmax_loss)
    print("Tmaxwell Relative L2: ", rel_L2_np(opt_tmax, true_tmax))
    print()
    print("True vx_shift")
    print(true_vxshift)
    print("Calculated vx_shift")
    print(opt_vxshift)
    print("vx_shift Loss: ", vxshift_loss)
    print("vx_shift Relative L2: ", rel_L2_np(opt_vxshift, true_vxshift))

    input()

    # print(maxwell_old[0])
    # print(maxwell[0])

    print("True Maxwell Sum:       ", np.sum(true_maxwell))
    print("Calculated Maxwell Sum: ", np.sum(opt_maxwell))
    print("Maxwell Loss: ", max_loss)
    print("Maxwell Relative L2: ", rel_L2_np(opt_maxwell, true_maxwell))
    input()



    print("Generating Images and Gifs")

    x = range(opt_tmax.size)
    generate_compare_plot(dir, "Tmaxwell", x, opt_tmax, x, true_tmax)
    x = range(opt_vxshift.size)
    generate_compare_plot(dir, "vx_shift", x, opt_vxshift, x, true_vxshift)

    generate_loss_plot(dir, "Loss", loss_list, xlabel="Epoch", ylabel="MSE Loss")

    tmax_gifgen.generate_gif()
    vx_shift_gifgen.generate_gif()