"""Training routines: base model, refit, replay, proximal-mu fine-tuning.

Every training run is fully determined by (data, torch_seed): the seed controls
network init (when not warm-started), the default train/val split, and minibatch
order.  A caller may instead supply explicit validation data; this is used by
the corrected paired experiment to share trajectory-level splits across rules.
Early stopping (predeclared in config_corrected.yaml): monitor the validation
objective after every epoch, keep the best checkpoint, stop after `patience` epochs
without improvement, restore the best checkpoint.  The validation objective is
val MSE plus, for the proximal rule, mu * ||theta - theta0||^2 (the same
objective the update rule itself minimizes).
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from model import DeltaMLP


def _l2sp(model, anchor):
    pen = 0.0
    for p, a in zip(model.parameters(), anchor):
        pen = pen + (p - a).pow(2).sum()
    return pen


def train_network(X, Y, cfg, torch_seed, init_state=None, mu=0.0,
                  val_data=None):
    """Train a DeltaMLP on transitions (X -> Y - X).  Returns
    (model, held-out one-step val MSE at the selected checkpoint, epochs run).

    init_state: warm start (proximal rule); also the L2-SP anchor when mu > 0.
    val_data: optional (X_val, Y_val) held-out data.  If supplied, X and Y are
        treated entirely as training data and no random within-array split is
        made.  This keeps the original experiment backward compatible while
        permitting shared, trajectory-level validation in corrected runs.
    """
    tc = cfg["training"]
    gen = torch.Generator().manual_seed(int(torch_seed))
    torch.manual_seed(int(torch_seed))

    model = DeltaMLP(tuple(cfg["model"]["hidden"]))
    if init_state is not None:
        model.load_state_dict(init_state)
    anchor = None
    if mu > 0.0:
        assert init_state is not None
        anchor = [torch.as_tensor(v).clone().detach()
                  for v in init_state.values() if v.dtype.is_floating_point]

    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    D_t = torch.as_tensor(np.asarray(Y) - np.asarray(X), dtype=torch.float32)
    n = X_t.shape[0]
    if val_data is None:
        n_val = max(1, int(round(tc["val_frac"] * n)))
        perm = torch.randperm(n, generator=gen)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        Xtr, Dtr = X_t[tr_idx], D_t[tr_idx]
        Xva, Dva = X_t[val_idx], D_t[val_idx]
    else:
        Xv, Yv = val_data
        Xtr, Dtr = X_t, D_t
        Xva = torch.as_tensor(np.asarray(Xv), dtype=torch.float32)
        Dva = torch.as_tensor(np.asarray(Yv) - np.asarray(Xv), dtype=torch.float32)
        if Xva.shape[0] == 0:
            raise ValueError("explicit validation data must be non-empty")
    n_tr = Xtr.shape[0]
    bs = min(tc["batch_size"], n_tr)

    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    mse = nn.MSELoss()

    best_obj, best_mse, best_state, best_epoch = np.inf, np.inf, None, -1
    epoch = -1
    for epoch in range(tc["max_epochs"]):
        model.train()
        order = torch.randperm(n_tr, generator=gen) if n_tr > bs else torch.arange(n_tr)
        for s in range(0, n_tr, bs):
            idx = order[s:s + bs]
            opt.zero_grad()
            loss = mse(model(Xtr[idx]), Dtr[idx])
            if mu > 0.0:
                loss = loss + mu * _l2sp(model, anchor)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_mse = mse(model(Xva), Dva).item()
            val_obj = val_mse + (mu * _l2sp(model, anchor).item() if mu > 0.0 else 0.0)
        if val_obj < best_obj:
            best_obj, best_mse, best_epoch = val_obj, val_mse, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= tc["patience"]:
            break
    model.load_state_dict(best_state)
    return model, best_mse, epoch + 1


def replay_mixture(X0, Y0, X1, Y1, rng):
    """D1 plus an equally sized random subsample of D0 (predeclared: without
    replacement if |D0| >= |D1|, else with replacement)."""
    n0, n1 = X0.shape[0], X1.shape[0]
    replace = n0 < n1
    idx = rng.choice(n0, size=n1, replace=replace)
    X = np.concatenate([X1, X0[idx]], axis=0)
    Y = np.concatenate([Y1, Y0[idx]], axis=0)
    return X, Y
