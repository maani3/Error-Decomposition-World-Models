"""Paired corrected rerun of the neural extension.

Corrections relative to a superseded independent-condition run:
  * one maximal D0 is generated per seed and N0 conditions are nested prefixes;
  * D1 and the refit model are shared across all N0 conditions for a seed;
  * validation holds out whole trajectories, not individual transitions;
  * all update rules see the same D1 train/validation trajectories;
  * proximal variants share minibatch randomness, isolating mu more cleanly;
  * outputs go only to results_corrected/ and never overwrite original results.
"""

import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import yaml

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)


def load_cfg():
    with open(os.path.join(BASE, "config_corrected.yaml")) as handle:
        return yaml.safe_load(handle)


def trajectory_ids(n_traj, transitions):
    """IDs matching generate_dataset's time-major flattening order."""
    return np.tile(np.arange(n_traj, dtype=np.int64), transitions)


def trajectory_split(ids):
    """Fixed 90/10 group split: trajectories 0,10,20,... are validation."""
    ids = np.asarray(ids)
    val = (ids % 10) == 0
    return ~val, val


def select_rows(X, Y, mask):
    return X[mask], Y[mask]


def int_seed(seed_sequence):
    return int(seed_sequence.generate_state(1)[0] % (2 ** 31))


def run_seed(cfg, seed):
    """Run all nested N0 conditions for one paired seed."""
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    from model import model_rollout
    from simulator import generate_dataset, make_panel
    from train import replay_mixture, train_network

    n0_grid = list(cfg["data"]["n0_grid"])
    n0_max = max(n0_grid)
    n1 = cfg["data"]["n1_trajectories"]
    transitions = cfg["data"]["traj_transitions"]

    root = np.random.SeedSequence([cfg["master_seed"], int(seed)])
    children = root.spawn(12)
    rng_d0 = np.random.default_rng(children[0])
    rng_d1 = np.random.default_rng(children[1])
    base_seed = int_seed(children[2])
    refit_seed = int_seed(children[3])
    replay_train_seed = int_seed(children[4])
    proximal_seed = int_seed(children[5])

    # Generate once: D0 conditions are literal prefixes and D1 is shared.
    X0_all, Y0_all = generate_dataset(cfg, "v0", n0_max, rng_d0)
    X1_all, Y1_all = generate_dataset(cfg, "v1", n1, rng_d1)
    ids0 = trajectory_ids(n0_max, transitions)
    ids1 = trajectory_ids(n1, transitions)
    d1_train_mask, d1_val_mask = trajectory_split(ids1)
    X1tr, Y1tr = select_rows(X1_all, Y1_all, d1_train_mask)
    X1va, Y1va = select_rows(X1_all, Y1_all, d1_val_mask)
    d1_val = (X1va, Y1va)

    panel = make_panel(cfg)
    horizons = cfg["horizons"]
    max_h = cfg["max_horizon"]
    theta_max = cfg["exclusions"]["divergence_theta_abs"]

    def panel_preds(model):
        full = model_rollout(model, panel, max_h)
        diverged = ((not np.all(np.isfinite(full)))
                    or np.any(np.abs(full[..., 0]) > theta_max))
        return full[horizons], bool(diverged)

    # Refit depends only on the shared D1, so train it exactly once per seed.
    refit, _, refit_epochs = train_network(
        X1tr, Y1tr, cfg, refit_seed, val_data=d1_val)
    refit_preds, refit_diverged = panel_preds(refit)

    records = []
    previous_prefix = set()
    for n0 in n0_grid:
        prefix_mask = ids0 < n0
        current_prefix = set(np.flatnonzero(prefix_mask).tolist())
        assert previous_prefix.issubset(current_prefix)
        previous_prefix = current_prefix

        d0_train_group, d0_val_group = trajectory_split(ids0)
        d0_train_mask = prefix_mask & d0_train_group
        d0_val_mask = prefix_mask & d0_val_group
        assert np.any(d0_train_mask) and np.any(d0_val_mask)
        assert not np.any(d0_train_mask & d0_val_mask)
        X0tr, Y0tr = select_rows(X0_all, Y0_all, d0_train_mask)
        X0va, Y0va = select_rows(X0_all, Y0_all, d0_val_mask)

        base, base_mse, base_epochs = train_network(
            X0tr, Y0tr, cfg, base_seed, val_data=(X0va, Y0va))
        base_state = base.state_dict()
        P0hat, base_diverged = panel_preds(base)

        P1hat = {"refit": refit_preds}
        epochs = {"base": base_epochs, "refit": refit_epochs}
        diverged = base_diverged or refit_diverged

        replay_rng = np.random.default_rng(
            np.random.SeedSequence([cfg["master_seed"], int(seed), int(n0), 991]))
        Xr, Yr = replay_mixture(X0tr, Y0tr, X1tr, Y1tr, replay_rng)
        replay, _, replay_epochs = train_network(
            Xr, Yr, cfg, replay_train_seed, val_data=d1_val)
        P1hat["replay"], replay_diverged = panel_preds(replay)
        epochs["replay"] = replay_epochs
        diverged = diverged or replay_diverged

        for mu in cfg["update_rules"]["proximal"]["mus"]:
            model, _, model_epochs = train_network(
                X1tr, Y1tr, cfg, proximal_seed, init_state=base_state,
                mu=float(mu), val_data=d1_val)
            name = f"proximal-{mu:g}"
            P1hat[name], model_diverged = panel_preds(model)
            epochs[name] = model_epochs
            diverged = diverged or model_diverged

        records.append({
            "n0": int(n0),
            "seed": int(seed),
            "base_val_mse": float(base_mse),
            "epochs": epochs,
            "diverged": bool(diverged),
            "P0hat": P0hat,
            "P1hat": P1hat,
            "split_counts": {
                "d0_train_trajectories": int(len(np.unique(ids0[d0_train_mask]))),
                "d0_val_trajectories": int(len(np.unique(ids0[d0_val_mask]))),
                "d1_train_trajectories": int(len(np.unique(ids1[d1_train_mask]))),
                "d1_val_trajectories": int(len(np.unique(ids1[d1_val_mask]))),
            },
        })

    # The shared refit must be bit-identical in every N0 record for this seed.
    assert all(np.array_equal(r["P1hat"]["refit"], refit_preds) for r in records)
    return records


def run_sweep(cfg, seeds, output_path, workers):
    if os.path.exists(output_path):
        raise FileExistsError(
            f"refusing to overwrite existing corrected result: {output_path}")
    started = time.time()
    records = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(run_seed, cfg, seed): seed for seed in seeds}
        for completed, future in enumerate(as_completed(futures), start=1):
            records.extend(future.result())
            if completed % 5 == 0 or completed == len(futures):
                elapsed = time.time() - started
                print(f"  {completed}/{len(futures)} paired seeds "
                      f"({elapsed:.0f}s elapsed)", flush=True)
    records.sort(key=lambda record: (record["n0"], record["seed"]))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump({"cfg": cfg, "records": records}, handle)
    print(f"Saved {len(records)} condition records -> {output_path}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    cfg = load_cfg()
    workers = min(10, os.cpu_count() or 1)
    output_dir = os.path.join(BASE, "results_corrected")
    if mode == "pilot":
        seeds = [0]
        output = os.path.join(output_dir, "pilot_v1.pkl")
    elif mode == "full":
        seeds = list(range(cfg["seeds_per_condition"]))
        output = os.path.join(BASE, cfg["corrected_design"]["results_file"])
    else:
        raise SystemExit(f"unknown mode: {mode}")
    print(f"Corrected {mode}: {len(seeds)} paired seeds on {workers} workers")
    run_sweep(cfg, seeds, output, workers=min(workers, len(seeds)))


if __name__ == "__main__":
    main()

