"""Unit tests (runnable directly: `python tests/test_units.py`, or via pytest).

(a) the identity eps1 = eps0 + (deltaM - deltaE) holds pointwise to numerical
    precision for actual trained networks;
(b) simulator energy behaves sensibly under damping (non-increasing);
(c) deltaE is nonzero on the whole panel under the g change.
"""

import copy
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import yaml  # noqa: E402

from simulator import (energy, env_panel_predictions, generate_dataset,  # noqa: E402
                       make_panel, rollout)
from model import model_rollout  # noqa: E402
from train import train_network  # noqa: E402


def load_cfg():
    with open(os.path.join(BASE, "config_corrected.yaml")) as f:
        return yaml.safe_load(f)


def test_identity_pointwise():
    """eps1 = eps0 + (deltaM - deltaE) for real trained nets, to fp precision."""
    cfg = load_cfg()
    cfg = copy.deepcopy(cfg)
    cfg["training"]["max_epochs"] = 60   # tiny budget: the identity is algebraic,
    cfg["training"]["patience"] = 20     # it must hold for ANY model
    rng = np.random.default_rng(123)
    X0, Y0 = generate_dataset(cfg, "v0", 10, rng)
    X1, Y1 = generate_dataset(cfg, "v1", 10, rng)
    m0, _, _ = train_network(X0, Y0, cfg, torch_seed=1)
    m1, _, _ = train_network(X1, Y1, cfg, torch_seed=2)

    panel = make_panel(cfg)
    hidx = np.array(cfg["horizons"])
    P0hat = model_rollout(m0, panel, cfg["max_horizon"])[hidx].reshape(len(hidx), -1)
    P1hat = model_rollout(m1, panel, cfg["max_horizon"])[hidx].reshape(len(hidx), -1)
    P0E = env_panel_predictions(cfg, "v0")[hidx].reshape(len(hidx), -1)
    P1E = env_panel_predictions(cfg, "v1")[hidx].reshape(len(hidx), -1)

    eps0, eps1 = P0hat - P0E, P1hat - P1E
    dM, dE = P1hat - P0hat, P1E - P0E
    resid = np.max(np.abs(eps1 - (eps0 + (dM - dE))))
    scale = max(1.0, float(np.max(np.abs(eps1))))
    assert resid <= 1e-12 * scale, f"identity residual {resid} too large"
    print(f"  identity residual: {resid:.3e} (pass, <= {1e-12 * scale:.1e})")


def test_energy_damped():
    """Total mechanical energy is non-increasing under damping (c > 0)."""
    cfg = load_cfg()
    v = cfg["simulator"]["v0"]
    x0 = np.array([[2.0, 1.5], [-1.0, -2.0], [0.25, 0.0], [3.0, 0.0]])
    traj = rollout(x0, 400, cfg["simulator"]["dt"], v["g"], v["L"], v["c"])
    E = energy(traj, v["g"], v["L"])  # (T+1, B)
    dE = np.diff(E, axis=0)
    assert np.all(dE <= 1e-10), f"energy increased by up to {dE.max()}"
    assert E[-1].max() < E[0].min() + 1e-9 or np.all(E[-1] < E[0]), \
        "energy did not decrease over the trajectory"
    print(f"  max energy increment: {dE.max():.3e} (pass, <= 1e-10); "
          f"E falls {E[0].mean():.2f} -> {E[-1].mean():.2f}")


def test_deltaE_nonzero():
    """deltaE is nonzero for every panel query at every horizon under the g change."""
    cfg = load_cfg()
    hidx = np.array(cfg["horizons"])
    P0E = env_panel_predictions(cfg, "v0")[hidx]
    P1E = env_panel_predictions(cfg, "v1")[hidx]
    per_query = np.linalg.norm(P1E - P0E, axis=2)  # (H, n_panel)
    thr = cfg["zero_effect_threshold"]
    n_zero = int((per_query < thr).sum())
    assert per_query.min() > thr, \
        f"{n_zero} zero-effect queries (min ||dE|| = {per_query.min():.3e})"
    print(f"  min per-query ||deltaE|| over panel x horizons: "
          f"{per_query.min():.3e} > {thr:g} (pass; 0 zero-effect queries)")


if __name__ == "__main__":
    print("test_identity_pointwise:")
    test_identity_pointwise()
    print("test_energy_damped:")
    test_energy_damped()
    print("test_deltaE_nonzero:")
    test_deltaE_nonzero()
    print("All tests passed.")
