import torch
import numpy as np

from test_functions import *

def rel_L2(pred, act, eps=1e-12):
    num = torch.linalg.norm(pred - act)
    den = torch.linalg.norm(pred)

    return num / (den + eps)

def rel_mse(pred, observed, eps=1e-8):
    return torch.mean(((pred - observed) / (observed + eps))**2)


torch.set_default_dtype(torch.float64)
np.set_printoptions(linewidth=150)

num_iters = 200
# patience = 40

norm_range = 20


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)
    if use_cuda:
        torch.cuda.manual_seed(72)
          
    # Known Goals results
    observed1 = torch.tensor(
        [307.86258, 487.92996, 289.91013, 328.09717, 391.98553],
        device=device, dtype=torch.float64
    )

    observed2 = torch.tensor(
        [235.59993, 75.7459,   99.17593,  200.80946, 128.5597],
        device=device, dtype=torch.float64
    )

    # Parameter bounds (example)
    x_min = torch.tensor(-norm_range, device=device, dtype=torch.float64)
    x_max = torch.tensor(norm_range, device=device, dtype=torch.float64)

    # Unknowns being differentiated
    initial_v = torch.nn.Parameter(torch.randn_like(observed1))
    initial_theta = torch.nn.Parameter(torch.randn_like(observed1))
    initial_m = torch.nn.Parameter(torch.randn_like(observed1))
    initial_c = torch.nn.Parameter(torch.randn_like(observed1))


    optimizer = torch.optim.Adam([initial_v, initial_theta, initial_m, initial_c], lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.75,
        patience=10,
        min_lr=1e-6
    )
    # optimizer = torch.optim.LBFGS([initial], lr=1.0)
    # loss_fun = lambda pred, observed: torch.mean((pred - observed)**2)

    best_loss = np.inf
    best_epoch = 0
    patience_ctr = 0
    for epoch in range(num_iters):

        # Get actual input from optimization values
        v = 3000.0 * torch.sigmoid(initial_v)
        theta = (torch.pi / 2) * torch.sigmoid(initial_theta)
        m = 50.0 * torch.sigmoid(initial_m) + 1e-3
        c = 1.0 * torch.sigmoid(initial_c)

        # Forward sim
        pred1, pred2 = simulate_projectile(v, theta, m, c)

        # Loss: match observed trajectory
        loss = (
            rel_mse(pred1, observed1) +
            rel_mse(pred2, observed2)
        )

        # Early Stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_preds = np.asarray([pred1.detach().cpu().numpy(), pred2.detach().cpu().numpy()])
            best_epoch = epoch
            patience_ctr = 0
        # else:
        #     patience_ctr += 1
        #     if patience_ctr >= patience:
        #         break


        # Backprop
        optimizer.zero_grad()
        loss.backward()

        # Clip Gradient
        torch.nn.utils.clip_grad_norm_([initial_v, initial_theta, initial_m, initial_c], max_norm=1.0)

        #Optimize
        optimizer.step()
        scheduler.step(loss)

        print(
            f"epoch: {epoch:<5} | "
            f"loss: {loss.item():<10.6e} | "
            f"learning rate: {optimizer.param_groups[0]['lr']:.2e}"
        )


    observations = np.asarray([observed1.detach().cpu().numpy(), observed2.detach().cpu().numpy()])
    print()
    print("Best Epoch: ", best_epoch)
    print("Best Loss: ", best_loss)
    print("Input:"
                f"\t v0:    {v.detach().cpu().numpy()}     \n"
                f"\t theta: {theta.detach().cpu().numpy()} \n"
                f"\t m:     {m.detach().cpu().numpy()}     \n"
                f"\t c:     {c.detach().cpu().numpy()}     \n"
    )
    print("Predicted:  \n", best_preds)
    print("Observed:   \n", observations)
    # print("L2 Error:   ", rel_L2(best_pred, observed).detach().cpu().numpy())