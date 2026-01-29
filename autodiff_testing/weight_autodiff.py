import torch
import numpy as np

from test_functions import *

def rel_L2(pred, act, eps=1e-12):
    num = torch.linalg.norm(pred - act)
    den = torch.linalg.norm(pred)

    return num / (den + eps)


num_iters = 10000
patience = 100

norm_range = 3000

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)
    if use_cuda:
        torch.cuda.manual_seed(72)
          
    # Unknown parameter
    observed = torch.tensor([1200.3, 1500.3, 1600.3, 1700.4, 1400.2], device=device, dtype=torch.float32)

    # Parameter bounds (example)
    x_min = torch.tensor(-norm_range, device=device, dtype=torch.float32)
    x_max = torch.tensor(norm_range, device=device, dtype=torch.float32)
    initial = torch.nn.Parameter(0.01 * torch.randn_like(observed, dtype=torch.float32))

    def normalize(x):
        return 2*(x-x_min)/(x_max - x_min) - 1

    def denormalize(x_norm):
        return x_min + (x_norm + 1.0) * 0.5 * (x_max - x_min)

    optimizer = torch.optim.Adam([initial], lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
    # optimizer = torch.optim.LBFGS([initial], lr=1.0)
    loss_fun = lambda pred, observed: torch.mean((pred - observed)**2)

    best_loss = np.inf
    best_epoch = 0
    patience_ctr = 0
    for epoch in range(num_iters):

        # Get actual input from optimization values
        weight = denormalize(torch.tanh(initial))

        # Forward simulation
        # pred = compute_weight(weight)
        pred = compute_weight(weight)

        # Loss: match observed trajectory
        loss = loss_fun(pred, observed)

        # Early Stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weight = weight
            best_pred = pred
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break


        # Backprop
        torch.nn.utils.clip_grad_norm_([initial], max_norm=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        print("epoch: ", epoch, "	loss: ", loss.item())

    print()
    print("Best Epoch: ", best_epoch)
    print("Input:      ", best_weight.detach().cpu().numpy())
    print("Predicted:  ", best_pred.detach().cpu().numpy())
    print("Observed:   ", observed.detach().cpu().numpy())
    print("L2 Error:   ", rel_L2(best_pred, observed).detach().cpu().numpy())
