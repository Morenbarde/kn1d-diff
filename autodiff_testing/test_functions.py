import torch

def compute_weight(masses):
    return masses*9.81


def iterative(values, max_iters=3):
    output = torch.ones(values.size(), device=values.device, dtype=values.dtype)

    for i in range(max_iters):
        output = output*values
        values = values - 100
    return output


def simulate_projectile(
    v0,        # initial speed
    theta,     # launch angle (radians)
    m,         # mass
    c,         # drag coefficient
    dt=0.01,
    steps=300,
    g=9.81
):
    """
    Differentiable projectile simulation with drag.
    Returns final (x, y) position.
    """

    # Initial velocity
    vx = v0 * torch.cos(theta)
    vy = v0 * torch.sin(theta)

    # Initial position
    x = torch.zeros(())
    y = torch.zeros(())

    for _ in range(steps):
        v = torch.sqrt(vx**2 + vy**2 + 1e-8)  # speed

        # Drag force (quadratic)
        Fx = -c * v * vx
        Fy = -c * v * vy - m * g

        # Acceleration
        ax = Fx / m
        ay = Fy / m

        # Euler integration
        vx = vx + ax * dt
        vy = vy + ay * dt

        x = x + vx * dt
        y = y + vy * dt

    return x, y