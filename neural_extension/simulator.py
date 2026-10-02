"""Versioned deterministic damped-pendulum simulator (the "world").

State x = (theta, omega).  Dynamics: theta_dot = omega,
omega_dot = -(g/L) sin(theta) - c * omega.  Integrated with RK4 at fixed dt.
All environment computations are float64 and noiseless; observation noise is
added only when building training datasets.
"""

import numpy as np


def deriv(x, g, L, c):
    """x: (..., 2) array -> (..., 2) array of time derivatives."""
    theta, omega = x[..., 0], x[..., 1]
    dtheta = omega
    domega = -(g / L) * np.sin(theta) - c * omega
    return np.stack([dtheta, domega], axis=-1)


def rk4_step(x, dt, g, L, c):
    k1 = deriv(x, g, L, c)
    k2 = deriv(x + 0.5 * dt * k1, g, L, c)
    k3 = deriv(x + 0.5 * dt * k2, g, L, c)
    k4 = deriv(x + dt * k3, g, L, c)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(x0, n_steps, dt, g, L, c):
    """Exact simulator rollout.  x0: (B, 2).  Returns (n_steps+1, B, 2)."""
    x0 = np.asarray(x0, dtype=np.float64)
    out = np.empty((n_steps + 1,) + x0.shape, dtype=np.float64)
    out[0] = x0
    x = x0
    for t in range(n_steps):
        x = rk4_step(x, dt, g, L, c)
        out[t + 1] = x
    return out


def energy(x, g, L):
    """Total mechanical energy (unit mass): 0.5*L^2*omega^2 + g*L*(1 - cos theta)."""
    theta, omega = x[..., 0], x[..., 1]
    return 0.5 * (L ** 2) * omega ** 2 + g * L * (1.0 - np.cos(theta))


def make_panel(cfg):
    """The predeclared query panel: all (theta, omega) grid points, fixed order."""
    states = [
        [th, om]
        for th in cfg["panel"]["thetas"]
        for om in cfg["panel"]["omegas"]
    ]
    return np.asarray(states, dtype=np.float64)


def env_panel_predictions(cfg, version):
    """P^v_E on the panel: (max_horizon+1, n_panel, 2) exact rollout."""
    v = cfg["simulator"][version]
    panel = make_panel(cfg)
    return rollout(panel, cfg["max_horizon"], cfg["simulator"]["dt"],
                   v["g"], v["L"], v["c"])


def generate_dataset(cfg, version, n_traj, rng):
    """Training trajectories with observation noise, split into transition pairs.

    Returns (X, Y): inputs x_t and targets x_{t+1}, both noisy, shape (N, 2).
    """
    v = cfg["simulator"][version]
    d = cfg["data"]
    T = d["traj_transitions"]
    th = rng.uniform(d["init_theta_range"][0], d["init_theta_range"][1], size=n_traj)
    om = rng.uniform(d["init_omega_range"][0], d["init_omega_range"][1], size=n_traj)
    x0 = np.stack([th, om], axis=-1)
    traj = rollout(x0, T, cfg["simulator"]["dt"], v["g"], v["L"], v["c"])  # (T+1, n, 2)
    noisy = traj + rng.normal(0.0, d["obs_noise_sigma"], size=traj.shape)
    X = noisy[:-1].reshape(-1, 2)
    Y = noisy[1:].reshape(-1, 2)
    return X, Y
