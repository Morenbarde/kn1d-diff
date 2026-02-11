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
OPTIMIZE_MESH_COEF = True
OPTIMIZE_COLLISION = True

# Learning Rate Parameters
INITIAL_LR = 1e-0
LR_CYCLE = 50
MIN_LR = 1e-6

# Iteration Parameters
NUM_ITERS = 50
CLIP_NORM = 1e-0

# Gif parameters
GENERATE_GIF = True
GIF_FPS = 10
GIF_FREQ = 5


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)
    # if use_cuda:
    #     torch.cuda.manual_seed(72)

    torch.set_default_dtype(torch.float64)
    torch.autograd.set_detect_anomaly(True)


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
    truein_A = torch.asarray(in_data["A"])
    print("A Range: ", torch.max(truein_A), torch.min(truein_A))
    truein_B = torch.asarray(in_data["B"])
    print("B Range: ", torch.max(truein_B), torch.min(truein_B))
    truein_C = torch.asarray(in_data["C"])
    print("C Range: ", torch.max(truein_C), torch.min(truein_C))
    truein_D = torch.asarray(in_data["D"])
    print("D Range: ", torch.max(truein_D), torch.min(truein_D))
    truein_F = torch.asarray(in_data["F"])
    print("F Range: ", torch.max(truein_F), torch.min(truein_F))
    truein_G = torch.asarray(in_data["G"])
    print("G Range: ", torch.max(truein_G), torch.min(truein_G))
    truein_CF_H_H = torch.asarray(in_data["CF_H_H"])
    print("CF_H_H Range: ", torch.max(truein_CF_H_H), torch.min(truein_CF_H_H))
    truein_CF_H_P = torch.asarray(in_data["CF_H_P"])
    print("CF_H_P Range: ", torch.max(truein_CF_H_P), torch.min(truein_CF_H_P))
    truein_CF_H_H2 = torch.asarray(in_data["CF_H_H2"])
    print("CF_H_H2 Range: ", torch.max(truein_CF_H_H2), torch.min(truein_CF_H_H2))
    # input()
    truein_meq_coeffs = MeshEqCoefficients(truein_A, truein_B, truein_C, truein_D, truein_F, truein_G)
    truein_collision_freqs = CollisionType(truein_CF_H_H, truein_CF_H_P, truein_CF_H_H2)


    # Desired Outputs
    
    trueout_fH = torch.asarray(out_data["fH"])
    trueout_Beta_CX_sum = torch.asarray(out_data["Beta_CX_sum"])
    trueout_Msum_H_H = torch.asarray(out_data["Msum_H_H"])
    trueout_Msum_H_P = torch.asarray(out_data["Msum_H_P"])
    trueout_Msum_H_H2 = torch.asarray(out_data["Msum_H_H2"])


    # --- Set up Kinetic_H ---
    
    h_mesh = KineticMesh('h', mesh_input["mu"], mesh_input["x"], mesh_input["Ti"], mesh_input["Te"], mesh_input["n"], mesh_input["PipeDia"], E0=mesh_input["E0"], fctr=mesh_input["fctr"], param_type='torch')
    kinetic_h = KineticH(h_mesh, torch.from_numpy(kh_in["mu"]), torch.from_numpy(kh_in["vxiA"]), torch.from_numpy(kh_in["fHBC"]), torch.from_numpy(kh_in["GammaxHBC"]), 
                        ni_correct=True, truncate=1.0e-3, max_gen=100, 
                        compute_errors=True, debrief=True, debug=False)

    # kinetic_h internal Data
    kinetic_h.debrief = False
    kinetic_h.H2_Moments.TH2 = torch.asarray(in_data['TH2_Moment'])
    kinetic_h.H2_Moments.VxH2 = torch.asarray(in_data['VxH2_Moment'])
    kinetic_h.Internal.fi_hat = torch.asarray(in_data['fi_hat'])
    kinetic_h.Internal.Alpha_CX = torch.asarray(in_data['Alpha_CX'])
    kinetic_h.Internal.ni = torch.asarray(in_data['ni'])
    kinetic_h.Internal.SIG_CX = torch.asarray(in_data['SIG_CX'])


    # --- Test Input Data ---

    # fH, Beta_CX_sum, m_sums = kinetic_h._run_generations(truein_fH, truein_meq_coeffs, truein_collision_freqs)
    # print("fH close: ", np.allclose(fH.detach(), trueout_fH))
    # print(rel_L2_torch(fH, trueout_fH))
    # print("Beta_CX_sum close: ", np.allclose(Beta_CX_sum.detach(), trueout_Beta_CX_sum))
    # print(rel_L2_torch(Beta_CX_sum, trueout_Beta_CX_sum))
    # print("Msum_H_H close: ", np.allclose(m_sums.H_H.detach(), trueout_Msum_H_H))
    # print(rel_L2_torch(m_sums.H_H, trueout_Msum_H_H))
    # print("Msum_H_P close: ", np.allclose(m_sums.H_P.detach(), trueout_Msum_H_P))
    # print(rel_L2_torch(m_sums.H_P, trueout_Msum_H_P))
    # print("Msum_H_H2 close: ", np.allclose(m_sums.H_H2.detach(), trueout_Msum_H_H2))
    # print(rel_L2_torch(m_sums.H_H2, trueout_Msum_H_H2))
    # input()


    # --- Optimization Parameters ---

    initial_fH = torch.nn.Parameter(torch.randn_like(truein_fH, requires_grad=True))
    # initial_fH = torch.nn.Parameter(torch.zeros_like(true_fH, requires_grad=True))
    initial_A = torch.nn.Parameter(torch.randn_like(truein_A, requires_grad=True))
    # initial_A = torch.nn.Parameter(torch.zeros_like(true_A, requires_grad=True))
    initial_B = torch.nn.Parameter(torch.randn_like(truein_B, requires_grad=True))
    # initial_B = torch.nn.Parameter(torch.zeros_like(true_B, requires_grad=True))
    initial_C = torch.nn.Parameter(torch.randn_like(truein_C, requires_grad=True))
    # initial_C = torch.nn.Parameter(torch.zeros_like(truein_C, requires_grad=True))
    initial_D = torch.nn.Parameter(torch.randn_like(truein_D, requires_grad=True))
    # initial_D = torch.nn.Parameter(torch.zeros_like(truein_D, requires_grad=True))
    initial_F = torch.nn.Parameter(torch.randn_like(truein_F, requires_grad=True))
    # initial_F = torch.nn.Parameter(torch.zeros_like(truein_F, requires_grad=True))
    initial_G = torch.nn.Parameter(torch.randn_like(truein_G, requires_grad=True))
    # initial_G = torch.nn.Parameter(torch.zeros_like(truein_G, requires_grad=True))
    initial_CF_H_H = torch.nn.Parameter(torch.randn_like(truein_CF_H_H, requires_grad=True))
    # initial_CF_H_H = torch.nn.Parameter(torch.zeros_like(truein_CF_H_H, requires_grad=True))
    initial_CF_H_P = torch.nn.Parameter(torch.randn_like(truein_CF_H_P, requires_grad=True))
    # initial_CF_H_HP = torch.nn.Parameter(torch.zeros_like(truein_CF_H_P, requires_grad=True))
    initial_CF_H_H2 = torch.nn.Parameter(torch.randn_like(truein_CF_H_H2, requires_grad=True))
    # initial_CF_H_H2 = torch.nn.Parameter(torch.zeros_like(truein_CF_H_H2, requires_grad=True))

    parameters = []
    if OPTIMIZE_FH:
        parameters.append(initial_fH)
    if OPTIMIZE_MESH_COEF:
        parameters.extend([initial_A, initial_B, initial_C, initial_D, initial_F, initial_G])
    if OPTIMIZE_COLLISION:
        parameters.extend([initial_CF_H_H, initial_CF_H_P, initial_CF_H_H2])

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
    if GENERATE_GIF:
        if OPTIMIZE_FH:
            fh_gifgen = GIF_Generator(NUM_ITERS, dir+"fH_Images/", "fH", truein_fH[5,10,:], fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_MESH_COEF:
            A_gifgen = GIF_Generator(NUM_ITERS, dir+"MEQ_Images/", "A", truein_A[5,10,:], fps=GIF_FPS, frequency=GIF_FREQ)
            B_gifgen = GIF_Generator(NUM_ITERS, dir+"MEQ_Images/", "B", truein_B[5,10,:], fps=GIF_FPS, frequency=GIF_FREQ)
            C_gifgen = GIF_Generator(NUM_ITERS, dir+"MEQ_Images/", "C", truein_C[5,10,:], fps=GIF_FPS, frequency=GIF_FREQ)
            D_gifgen = GIF_Generator(NUM_ITERS, dir+"MEQ_Images/", "D", truein_D[5,10,:], fps=GIF_FPS, frequency=GIF_FREQ)
            F_gifgen = GIF_Generator(NUM_ITERS, dir+"MEQ_Images/", "F", truein_F[5,10,:], fps=GIF_FPS, frequency=GIF_FREQ)
            G_gifgen = GIF_Generator(NUM_ITERS, dir+"MEQ_Images/", "G", truein_G[5,10,:], fps=GIF_FPS, frequency=GIF_FREQ)
        if OPTIMIZE_COLLISION:
            H_H_gifgen = GIF_Generator(NUM_ITERS, dir+"Collision_Frequency_Images/", "H_H", truein_CF_H_H, fps=GIF_FPS, frequency=GIF_FREQ)
            H_P_gifgen = GIF_Generator(NUM_ITERS, dir+"Collision_Frequency_Images/", "H_P", truein_CF_H_P, fps=GIF_FPS, frequency=GIF_FREQ)
            H_H2_gifgen = GIF_Generator(NUM_ITERS, dir+"Collision_Frequency_Images/", "H_H2", truein_CF_H_H2, fps=GIF_FPS, frequency=GIF_FREQ)


    # Capture Best Epoch
    loss_list = []
    lr_list = []
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(NUM_ITERS):

        # --- Bound Inputs ---

        if OPTIMIZE_FH:
            # fH = 1e19 * torch.nn.functional.softplus(initial_fH)
            # fH = 1e19*torch.nn.functional.tanh(initial_fH)
            fH = 1e19 * torch.sigmoid(initial_fH)
            # fH = initial_fH
        else:
            fH = truein_fH

        if OPTIMIZE_MESH_COEF:
            mA = torch.sigmoid(initial_A)
            mB = torch.sigmoid(initial_B) 
            mC = torch.sigmoid(initial_C)
            mD = torch.sigmoid(initial_D) 
            mF = 1e18 * torch.sigmoid(initial_F) 
            # mF = 1e18 * torch.tanh(initial_F) 
            mG = 1e18 * torch.sigmoid(initial_G)
            # mG = 1e18 * torch.tanh(initial_G) 
            meq_coeffs = MeshEqCoefficients(mA, mB, mC, mD, mF, mG)
        else:
            meq_coeffs = truein_meq_coeffs

        if OPTIMIZE_COLLISION:
            cf_hh = 1e-2 * torch.sigmoid(initial_CF_H_H)
            cf_hp = 1e+1 * torch.sigmoid(initial_CF_H_P)
            cf_hh2 = 1e-1 * torch.sigmoid(initial_CF_H_H2)
            coll_freqs = CollisionType(cf_hh, cf_hp, cf_hh2)
        else:
            coll_freqs = truein_collision_freqs


        # --- Run Function ---
        fH, Beta_CX_sum, m_sums = kinetic_h._run_generations(fH, meq_coeffs, coll_freqs)


        # --- Optimize ---

        # Compute Loss
        loss1 = loss_fun(fH, trueout_fH)
        loss2 = loss_fun(Beta_CX_sum, trueout_Beta_CX_sum)
        loss3 = loss_fun(m_sums.H_H, trueout_Msum_H_H)
        loss4 = loss_fun(m_sums.H_P, trueout_Msum_H_P)
        loss5 = loss_fun(m_sums.H_H2, trueout_Msum_H_H2)
        
        loss = loss1 + loss2 + loss3 + loss4 + loss5

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
            best_inputs = [
                        fH.detach().cpu(),
                        MeshEqCoefficients(
                            mA.detach().cpu(),
                            mB.detach().cpu(),
                            mC.detach().cpu(),
                            mD.detach().cpu(),
                            mF.detach().cpu(),
                            mG.detach().cpu()
                            ),
                        CollisionType(
                            cf_hh.detach().cpu(),
                            cf_hp.detach().cpu(),
                            cf_hh2.detach().cpu()
                            )
                        ]
            
            best_pred = [fH.detach().cpu(), Beta_CX_sum.detach().cpu(), CollisionType(m_sums.H_H.detach().cpu(), m_sums.H_P.detach().cpu(), m_sums.H_H2.detach().cpu())]
            best_epoch = epoch

        print(
            f"epoch: {epoch:<5} | "
            f"loss: {loss.item():<10.6e} | "
            # f"learning rate: {optimizer.param_groups[0]['lr']:.2e}"
            f"learning rate: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Update Gif data
        if GENERATE_GIF:
            if OPTIMIZE_FH:
                fh_gifgen.update(fH[5,10,:], epoch)
            if OPTIMIZE_MESH_COEF:
                A_gifgen.update(mA[5,10,:], epoch)
                B_gifgen.update(mB[5,10,:], epoch)
                C_gifgen.update(mC[5,10,:], epoch)
                D_gifgen.update(mD[5,10,:], epoch)
                F_gifgen.update(mF[5,10,:], epoch)
                G_gifgen.update(mG[5,10,:], epoch)
            if OPTIMIZE_COLLISION:
                H_H_gifgen.update(cf_hh, epoch)
                H_P_gifgen.update(cf_hp, epoch)
                H_H2_gifgen.update(cf_hh2, epoch)








    # --- Analysis ---

    opt_fH_in, opt_me_coeffs, opt_coll_freq = best_inputs[0], best_inputs[1], best_inputs[2]
    opt_fH_out, opt_Beta_CX_sum, opt_m_sums = best_pred[0], best_pred[1], best_pred[2]

    # --- Analyze ---
    print("Best Epoch: ", best_epoch)

    # Optimized Inputs Analysis
    if OPTIMIZE_FH:
        fH_in_loss = loss_fun(opt_fH_in, truein_fH).item()
        print("fH Input Loss: ", fH_in_loss)
        print("fH Input Relative L2: ", rel_L2_torch(opt_fH_in, truein_fH))
        print()

    if OPTIMIZE_MESH_COEF:
        A_loss = loss_fun(opt_me_coeffs.A, truein_A).item()
        B_loss = loss_fun(opt_me_coeffs.B, truein_B).item()
        C_loss = loss_fun(opt_me_coeffs.C, truein_C).item()
        D_loss = loss_fun(opt_me_coeffs.D, truein_D).item()
        F_loss = loss_fun(opt_me_coeffs.F, truein_F).item()
        G_loss = loss_fun(opt_me_coeffs.G, truein_G).item()
        print("Mesh Coef Loss: ", A_loss, B_loss, C_loss, D_loss, F_loss, G_loss)
        A_l2 = rel_L2_torch(opt_me_coeffs.A, truein_A).item()
        B_l2 = rel_L2_torch(opt_me_coeffs.B, truein_B).item()
        C_l2 = rel_L2_torch(opt_me_coeffs.C, truein_C).item()
        D_l2 = rel_L2_torch(opt_me_coeffs.D, truein_D).item()
        F_l2 = rel_L2_torch(opt_me_coeffs.F, truein_F).item()
        G_l2 = rel_L2_torch(opt_me_coeffs.G, truein_G).item()
        print("Mesh Coef Relative L2: ", A_l2, B_l2, C_l2, D_l2, F_l2, G_l2)
        print()

    if OPTIMIZE_COLLISION:
        CF_H_H_loss = rel_L2_torch(opt_coll_freq.H_H, truein_CF_H_H).item()
        CF_H_P_loss = rel_L2_torch(opt_coll_freq.H_P, truein_CF_H_P).item()
        CF_H_H2_loss = rel_L2_torch(opt_coll_freq.H_H2, truein_CF_H_H2).item()
        print("Collision Loss: ", CF_H_H_loss, CF_H_P_loss, CF_H_H2_loss)
        CF_H_H_l2 = rel_L2_torch(opt_coll_freq.H_H, truein_CF_H_H).item()
        CF_H_P_l2 = rel_L2_torch(opt_coll_freq.H_P, truein_CF_H_P).item()
        CF_H_H2_l2 = rel_L2_torch(opt_coll_freq.H_H2, truein_CF_H_H2).item()
        print("Collision Loss: ", CF_H_H_loss, CF_H_P_loss, CF_H_H2_loss)
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
    msum_H_H_loss = rel_L2_torch(opt_coll_freq.H_H, trueout_Msum_H_H).item()
    msum_H_P_loss = rel_L2_torch(opt_coll_freq.H_P, trueout_Msum_H_P).item()
    msum_H_H2_loss = rel_L2_torch(opt_coll_freq.H_H2, trueout_Msum_H_H2).item()
    print("M_Sum Loss: ", msum_H_H_loss, msum_H_P_loss, msum_H_H2_loss)
    msum_H_H_l2 = rel_L2_torch(opt_coll_freq.H_H, trueout_Msum_H_H).item()
    msum_H_P_l2 = rel_L2_torch(opt_coll_freq.H_P, trueout_Msum_H_P).item()
    msum_H_H2_l2 = rel_L2_torch(opt_coll_freq.H_H2, trueout_Msum_H_H2).item()
    print("M_Sum Relative L2: ", msum_H_H_loss, msum_H_P_loss, msum_H_H2_loss)
    print()


    # --- Plot Generation --- 

    print("Generating Images and Gifs")

    # Runtime Data
    generate_loss_plot(dir, "Loss", loss_list, xlabel="Epoch", ylabel="Symmetrical Loss")
    generate_lr_plot(dir, "LR", lr_list, xlabel="Epoch", ylabel="Learning Rate")

    # fH
    x = range(opt_fH_in[5,10,:].numel())
    for i in range(len(opt_fH_in[5,:,0])):
        generate_compare_plot(dir+"FH_Results/", "fH"+str(i), x, opt_fH_in[5,i,:], x, truein_fH[5,i,:])

    # MEQ Coeffs
    x = range(opt_me_coeffs.A[5,10,:].numel())
    for i in range(len(opt_me_coeffs.A[5,:,0])):
        generate_compare_plot(dir+"MEQ_Results/", "A"+str(i), x, opt_me_coeffs.A[5,i,:], x, truein_A[5,i,:])
        generate_compare_plot(dir+"MEQ_Results/", "B"+str(i), x, opt_me_coeffs.B[5,i,:], x, truein_B[5,i,:])
        generate_compare_plot(dir+"MEQ_Results/", "C"+str(i), x, opt_me_coeffs.C[5,i,:], x, truein_C[5,i,:])
        generate_compare_plot(dir+"MEQ_Results/", "D"+str(i), x, opt_me_coeffs.D[5,i,:], x, truein_D[5,i,:])
        generate_compare_plot(dir+"MEQ_Results/", "E"+str(i), x, opt_me_coeffs.F[5,i,:], x, truein_F[5,i,:])
        generate_compare_plot(dir+"MEQ_Results/", "F"+str(i), x, opt_me_coeffs.G[5,i,:], x, truein_G[5,i,:])

    # Collision Frequencies
    x = range(opt_coll_freq.H_H.numel())
    for i in range(opt_coll_freq.H_H.numel()):
        generate_compare_plot(dir+"Collision_Frequency_Results/", "H_H"+str(i), x, opt_coll_freq.H_H, x, truein_CF_H_H)
        generate_compare_plot(dir+"Collision_Frequency_Results/", "H_P"+str(i), x, opt_coll_freq.H_P, x, truein_CF_H_P)
        generate_compare_plot(dir+"Collision_Frequency_Results/", "H_H2"+str(i), x, opt_coll_freq.H_H2, x, truein_CF_H_H2)

    # --- Gif Generation ---
    if GENERATE_GIF:
        if OPTIMIZE_FH:
            fh_gifgen.generate_gif()

    if GENERATE_GIF:
        if OPTIMIZE_FH:
            fh_gifgen.generate_gif()
        if OPTIMIZE_MESH_COEF:
            A_gifgen.generate_gif()
            B_gifgen.generate_gif()
            C_gifgen.generate_gif()
            D_gifgen.generate_gif()
            F_gifgen.generate_gif()
            G_gifgen.generate_gif()
        if OPTIMIZE_COLLISION:
            H_H_gifgen.generate_gif()
            H_P_gifgen.generate_gif()
            H_H2_gifgen.generate_gif()