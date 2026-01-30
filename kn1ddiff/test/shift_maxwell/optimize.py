import torch
import numpy as np

from kn1ddiff.create_shifted_maxwellian import *




data_file = "in_out2.npz"
num_iters = 500


if __name__ == "__main__":
    
    data = np.load("kn1ddiff/test/shift_maxwell/"+data_file)
    print(data["mol"])
    print(data["mu"])


    # --- Load Inputs and Outputs ---

    # Fixed
    vx = torch.from_numpy(data["vx"])
    vr = torch.from_numpy(data["vr"])
    mu = torch.from_numpy(data["mu"])
    mol = torch.from_numpy(data["mol"])
    Tnorm = torch.from_numpy(data["Tnorm"])

    # Gradient
    Tmaxwell = torch.from_numpy(data["Tmaxwell"])
    vx_shift = torch.from_numpy(data["vx_shift"])

    # Desired Outputs
    maxwell_old = torch.from_numpy(data["maxwell"])
    # with torch.no_grad():
    #     maxwell_old = create_shifted_maxwellian(vr, vx, Tmaxwell, vx_shift, mu, mol, Tnorm)


    # --- Test Optimization ---
    initial_tmaxwell = torch.nn.Parameter(torch.randn_like(Tmaxwell))
    initial_vx_shift = torch.nn.Parameter(torch.randn_like(vx_shift))

    optimizer = torch.optim.Adam([initial_tmaxwell, initial_vx_shift], lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=30,
        min_lr=1e-8
    )

    loss_fun = torch.nn.MSELoss()

    for epoch in range(num_iters):
        tmax = 600 * torch.sigmoid(initial_tmaxwell)
        shift = 5000 * torch.sigmoid(initial_vx_shift)
        # tmax = 1 * torch.nn.functional.softplus(initial_tmaxwell)
        # shift = 5e3 * torch.nn.functional.softplus(initial_vx_shift)

        maxwell = create_shifted_maxwellian(vr, vx, tmax, shift, mu, mol, Tnorm)

        # Compute Loss
        loss = loss_fun(maxwell, maxwell_old)


        # Backprop
        optimizer.zero_grad()
        loss.backward()

        # Clip Gradient
        torch.nn.utils.clip_grad_norm_([initial_tmaxwell, initial_vx_shift], max_norm=1.0)

        #Optimize
        optimizer.step()
        scheduler.step(loss)

        print(
            f"epoch: {epoch:<5} | "
            f"loss: {loss.item():<10.6e} | "
            f"learning rate: {optimizer.param_groups[0]['lr']:.2e}"
        )
        # print(initial_tmaxwell.grad)
        # print(initial_vx_shift.grad)
        # input()

    print(torch.min(Tmaxwell), torch.max(Tmaxwell))
    print(torch.min(vx_shift), torch.max(vx_shift))
    input()

    print(Tmaxwell)
    print(tmax)
    print()
    print(vx_shift)
    print(shift)
    input()

    print(maxwell_old[0])
    print(maxwell[0])

    print(torch.sum(maxwell_old[0,:,:]))
    print(torch.sum(maxwell[0,:,:]))

