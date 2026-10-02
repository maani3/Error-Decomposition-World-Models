"""
Second-pass experiments E5-E10 for
"Grading the Update, Not the Model: Decomposing Error in Continual World Models".

E5  K-round accumulation      -- does per-round mismatch predict K-round endpoint
                                 better than per-round endpoint does?  (the paper's
                                 central justification, previously only argued)
E6  Nonlinear stress test     -- does the disagreement survive nonlinear dynamics,
                                 where the parallel-trends assumption is threatened?
E7  Aggregation               -- worst-case vs global error over interventions
E8  Wilson CIs                -- uncertainty on the headline disagreement rates
E9  Non-oracle ablation       -- does a proximal update also reverse the ranking?
E10 Cancellation geometry     -- how do inherited error and mismatch combine?

    python3 followup_experiments.py
"""
import json
import numpy as np

D, SIGMA = 5, 0.25
P   = lambda M, h: np.linalg.matrix_power(M, h)
rho = lambda M: abs(np.linalg.eigvals(M)).max()
rel = lambda X, Y: np.linalg.norm(X - Y) / np.linalg.norm(Y)

def base_world(rng, target_rho=0.70):
    A = rng.normal(0, .45, (D, D)) * (rng.random((D, D)) < .55)
    r = rho(A)
    if r < 1e-6: return None
    return A * (target_rho / r)

def samp(A, n, rng, nonlin=False):
    X = rng.normal(0, 1, (D, n))
    Y = np.tanh(A @ X) if nonlin else A @ X
    return X, Y + rng.normal(0, SIGMA, (D, n))

def ols(X, Y, lam=0.):
    return Y @ X.T @ np.linalg.inv(X @ X.T + lam * np.eye(D))

def proximal(Aprev, X, Y, mu):
    """Realistic non-oracle continual update: least squares anchored to the previous model."""
    return (Y @ X.T + mu * Aprev) @ np.linalg.inv(X @ X.T + mu * np.eye(D))

def oracle_entry(Aprev, X, Y, cell):
    r, c = cell
    A = Aprev.copy()
    resid = Y[r] - Aprev[r] @ X + Aprev[r, c] * X[c]
    A[r, c] = resid @ X[c] / (X[c] @ X[c])
    return A

def wilson_ci(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return max(0., center - half), min(1., center + half)

# ============================================================ E5 ==============
def E5(seeds=400, K=10, n_round=25, n_base=400, mu=30., lam_base=40., H=(1, 5)):
    """Sequential regime: one mechanism entry changes per round, limited data per round."""
    rules = ("refit", "replay", "proximal", "adapter")
    endpoint = {r: {h: np.full((seeds, K + 1), np.nan) for h in H} for r in rules}
    mismatch = {r: {h: np.full((seeds, K), np.nan) for h in H} for r in rules}
    used = 0
    for s in range(seeds):
        rng = np.random.default_rng(10_000 + s)
        A = base_world(rng)
        if A is None: continue
        # base model, fit on plentiful pre-change data, ridge-shrunk -> inherited bias
        Xb, Yb = samp(A, n_base, rng)
        M0 = ols(Xb, Yb, lam_base)
        cur = {r: M0.copy() for r in rules}
        buf_X, buf_Y = [Xb], [Yb]
        A_true = A.copy()
        ok = True
        for r_ in rules:
            for h in H: endpoint[r_][h][s, 0] = rel(P(cur[r_], h), P(A_true, h))
        for k in range(K):
            # --- world changes one entry -------------------------------------
            for _ in range(50):
                cell = (rng.integers(D), rng.integers(D))
                cand = A_true.copy(); cand[cell] += rng.choice([-1, 1]) * 0.25
                if rho(cand) < 0.95: break
            else:
                ok = False; break
            A_prev_true, A_true = A_true, cand
            Xk, Yk = samp(A_true, n_round, rng)
            buf_X.append(Xk); buf_Y.append(Yk)
            prev = {r_: cur[r_].copy() for r_ in rules}
            cur["refit"]    = ols(Xk, Yk, 1e-3)
            cur["replay"]   = ols(np.hstack(buf_X), np.hstack(buf_Y), 1e-3)
            cur["proximal"] = proximal(prev["proximal"], Xk, Yk, mu)
            cur["adapter"]  = oracle_entry(prev["adapter"], Xk, Yk, cell)
            for h in H:
                dE = P(A_true, h) - P(A_prev_true, h)
                nE = np.linalg.norm(dE)
                for r_ in rules:
                    endpoint[r_][h][s, k + 1] = rel(P(cur[r_], h), P(A_true, h))
                    if nE > 1e-9:
                        dM = P(cur[r_], h) - P(prev[r_], h)
                        mismatch[r_][h][s, k] = np.linalg.norm(dM - dE) / nE
        if ok: used += 1
        else:
            for r_ in rules:
                for h in H:
                    endpoint[r_][h][s, :] = np.nan; mismatch[r_][h][s, :] = np.nan
    out = {"used": used, "K": K, "n_round": n_round, "rules": list(rules)}
    for h in H:
        out[f"endpoint_h{h}"] = {r: np.nanmedian(endpoint[r][h], 0).tolist() for r in rules}
        out[f"mismatch_h{h}"] = {r: float(np.nanmedian(mismatch[r][h][:, 0])) for r in rules}
        out[f"mismatch_mean_h{h}"] = {r: float(np.nanmedian(np.nanmean(mismatch[r][h], 1))) for r in rules}
        out[f"round1_endpoint_h{h}"] = {r: float(np.nanmedian(endpoint[r][h][:, 1])) for r in rules}
        out[f"final_endpoint_h{h}"]  = {r: float(np.nanmedian(endpoint[r][h][:, K])) for r in rules}

        # Recovering the final rule ordering per seed is a more direct test than
        # comparing rankings of four cross-seed medians.
        final_matrix = np.column_stack([endpoint[r][h][:, K] for r in rules])
        predictors = {
            "round1_endpoint": np.column_stack([endpoint[r][h][:, 1] for r in rules]),
            "round1_mismatch": np.column_stack([mismatch[r][h][:, 0] for r in rules]),
            "mean_mismatch": np.column_stack(
                [np.nanmean(mismatch[r][h], axis=1) for r in rules]),
        }

        def ranking_diagnostics(pred, target):
            valid = np.all(np.isfinite(pred), axis=1) & np.all(
                np.isfinite(target), axis=1)
            pred, target = pred[valid], target[valid]
            pair_by_seed, spearman = [], []
            for p, t in zip(pred, target):
                seed_hits = []
                for i in range(len(rules)):
                    for j in range(i + 1, len(rules)):
                        dp, dt = p[i] - p[j], t[i] - t[j]
                        seed_hits.append((dp == 0 and dt == 0) or dp * dt > 0)
                pair_by_seed.append(np.mean(seed_hits))
                rp = np.empty(len(rules), dtype=float)
                rt = np.empty(len(rules), dtype=float)
                rp[np.argsort(p)] = np.arange(len(rules))
                rt[np.argsort(t)] = np.arange(len(rules))
                spearman.append(float(np.corrcoef(rp, rt)[0, 1]))
            pair_by_seed, spearman = np.array(pair_by_seed), np.array(spearman)
            summary = {
                "n_seeds": int(valid.sum()),
                "pairwise_order_accuracy": float(np.mean(pair_by_seed)),
                "best_rule_accuracy": float(
                    np.mean(np.argmin(pred, axis=1) == np.argmin(target, axis=1))
                ),
                "mean_spearman": float(np.mean(spearman)),
                "median_spearman": float(np.median(spearman)),
            }
            return summary, pair_by_seed, spearman

        summaries, raw_pair, raw_rho = {}, {}, {}
        for name, values in predictors.items():
            summaries[name], raw_pair[name], raw_rho[name] = ranking_diagnostics(
                values, final_matrix)
        out[f"agreement_with_final_h{h}"] = summaries

        # Paired seed bootstrap: how much more informative is round-1 mismatch
        # than round-1 endpoint for the final ten-round rule ordering?
        rng_boot = np.random.default_rng(30_000 + h)
        n_valid = len(raw_pair["round1_endpoint"])
        idx = rng_boot.integers(0, n_valid, size=(10_000, n_valid))
        pair_diff = (raw_pair["round1_mismatch"] -
                     raw_pair["round1_endpoint"])
        rho_diff = (raw_rho["round1_mismatch"] -
                    raw_rho["round1_endpoint"])
        out[f"mismatch_advantage_h{h}"] = {
            "pairwise_accuracy_difference": float(np.mean(pair_diff)),
            "pairwise_difference_ci95": np.percentile(
                pair_diff[idx].mean(axis=1), [2.5, 97.5]).tolist(),
            "mean_spearman_difference": float(np.mean(rho_diff)),
            "spearman_difference_ci95": np.percentile(
                rho_diff[idx].mean(axis=1), [2.5, 97.5]).tolist(),
        }
    return out

# ============================================================ E6 ==============
def E6(seeds=400, H=8, bump=0.35, lam=15., n0=400, n1=400, sigma=0.10):
    """Nonlinear dynamics  x_{t+1} = A tanh(x_t).

    Chosen so that ESTIMATION stays linear in the parameters -- y is linear in
    tanh(x), so plain OLS on (tanh(x), y) is exact and unbiased -- while the
    h-step ROLLOUT is genuinely nonlinear (A composed with tanh, h times).
    This isolates the effect of nonlinear composition from any estimator
    handicap.  An earlier version fitted via arctanh(y), which amplifies noise
    near +-1 and handicapped `refit` for reasons unrelated to nonlinearity.
    """
    def roll(A, h):
        Xs = np.eye(D)
        for _ in range(h):
            Xs = A @ np.tanh(Xs)
        return Xs
    def nsamp(A, n, rng):
        X = rng.normal(0, 1, (D, n))
        return np.tanh(X), A @ np.tanh(X) + rng.normal(0, sigma, (D, n))
    rules = ("refit", "replay", "adapter")
    res = {r: {"endpoint": [], "mismatch": []} for r in rules}
    inherit, used, att = [], 0, 0
    for s_ in range(seeds):
        att += 1
        rng = np.random.default_rng(20_000 + s_)
        A0 = base_world(rng)
        if A0 is None: continue
        A1 = A0.copy(); A1[2, 1] += bump
        Fp, Yp = nsamp(A0, n0, rng)
        Fq, Yq = nsamp(A1, n1, rng)
        M0 = ols(Fp, Yp, lam)
        cand = {"refit":   ols(Fq, Yq, 0.),
                "replay":  ols(np.hstack([Fp[:, -n1:], Fq]), np.hstack([Yp[:, -n1:], Yq]), 0.),
                "adapter": oracle_entry(M0, Fq, Yq, (2, 1))}
        dEn = np.array([np.linalg.norm(roll(A1, h) - roll(A0, h)) for h in range(1, H + 1)])
        if dEn.min() < 1e-9 or not np.all(np.isfinite(dEn)): continue
        inh = np.array([rel(roll(M0, h), roll(A0, h)) for h in range(1, H + 1)])
        cell, good = {}, True
        for k, M1 in cand.items():
            ep = np.array([rel(roll(M1, h), roll(A1, h)) for h in range(1, H + 1)])
            mm = np.array([np.linalg.norm((roll(M1, h) - roll(M0, h)) - (roll(A1, h) - roll(A0, h)))
                           / dEn[h - 1] for h in range(1, H + 1)])
            if not (np.all(np.isfinite(ep)) and np.all(np.isfinite(mm))): good = False
            cell[k] = (ep, mm)
        if not good or not np.all(np.isfinite(inh)): continue
        used += 1; inherit.append(inh)
        for k in rules:
            res[k]["endpoint"].append(cell[k][0]); res[k]["mismatch"].append(cell[k][1])
    ea = np.array(res["refit"]["endpoint"]); eb = np.array(res["adapter"]["endpoint"])
    ma = np.array(res["refit"]["mismatch"]); mb = np.array(res["adapter"]["mismatch"])
    dis = np.mean((ea < eb) & (mb < ma), axis=0)
    return {"attempted": att, "used": used, "H": H,
            "inherited_h1": float(np.median(np.array(inherit)[:, 0])),
            "disagreement": dis.tolist(),
            "median": {k: {m: np.median(np.array(res[k][m]), 0).tolist()
                           for m in ("endpoint", "mismatch")} for k in rules}}

# ============================================================ E7 ==============
def E7(seeds=600, H=12, lam=40., n0=400, n1=400, bump=0.35):
    """Worst-case column error vs global Frobenius error.

    Both use a single matrix-level denominator.  This keeps zero-effect
    intervention columns in the numerator instead of silently discarding
    unintended changes on them.
    """
    rules = ("refit", "replay", "adapter")
    acc = {r: {"global": [], "worst": []} for r in rules}
    used = 0
    for s in range(seeds):
        rng = np.random.default_rng(s)
        A0 = base_world(rng)
        if A0 is None: continue
        A1 = A0.copy(); A1[2, 1] += bump
        if rho(A1) > 0.82: continue
        Xp, Yp = samp(A0, n0, rng); M0 = ols(Xp, Yp, lam)
        Xq, Yq = samp(A1, n1, rng)
        cand = {"refit":   ols(Xq, Yq),
                "replay":  ols(np.hstack([Xp[:, -n1:], Xq]), np.hstack([Yp[:, -n1:], Yq])),
                "adapter": oracle_entry(M0, Xq, Yq, (2, 1))}
        global_c, worst_c, good = {}, {}, True
        for k, M1 in cand.items():
            global_, worst_ = [], []
            for h in range(1, H + 1):
                dE = P(A1, h) - P(A0, h); dM = P(M1, h) - P(M0, h)
                denom_global = np.linalg.norm(dE, "fro")
                denom_worst = np.linalg.norm(dE, axis=0).max()
                if denom_global <= 1e-12 or denom_worst <= 1e-12:
                    good = False; break
                global_.append(np.linalg.norm(dM - dE, "fro") / denom_global)
                worst_.append(np.linalg.norm(dM - dE, axis=0).max() / denom_worst)
                if not np.isfinite(global_[-1]) or not np.isfinite(worst_[-1]):
                    good = False; break
            if not good: break
            global_c[k], worst_c[k] = np.array(global_), np.array(worst_)
        if not good: continue
        used += 1
        for k in rules:
            acc[k]["global"].append(global_c[k])
            acc[k]["worst"].append(worst_c[k])
    out = {"used": used, "H": H}
    for k in rules:
        g_ = np.median(np.array(acc[k]["global"]), 0)
        w_ = np.median(np.array(acc[k]["worst"]), 0)
        out[k] = {"global": g_.tolist(), "worst": w_.tolist(),
                  "ratio_h1": float(w_[0] / g_[0])}
    return out

# ============================================================ E8 ==============
def E8(path="results.json"):
    """Wilson 95% CIs on the headline disagreement rates (E2/E3)."""
    d = json.load(open(path))
    out = {}
    for cond in ("A_poor", "A_good"):
        n = d[cond]["usable"]
        out[cond] = {}
        for hi, h in ((0, 1), (4, 5), (11, 12)):
            p = d[cond]["disagreement"][hi]
            k = round(p * n)
            lo, hiq = wilson_ci(k, n)
            out[cond][f"h{h}"] = {
                "count": int(k), "p": k / n, "n": n,
                "ci95": [float(lo), float(hiq)]}
    return out

# ============================================================ E9 ==============
def E9(seeds=600, mus=(1., 5., 10., 20., 40., 80., 160.),
       horizons=(1, 5, 12), lam=40., n0=400, n1=400, bump=0.35):
    """Single-round non-oracle proximal sweep against post-change refitting."""
    acc = {mu: {h: {"refit_ep": [], "refit_mm": [],
                    "prox_ep": [], "prox_mm": [], "reverse": []}
                for h in horizons} for mu in mus}
    used = 0
    for s in range(seeds):
        rng = np.random.default_rng(s)
        A0 = base_world(rng)
        if A0 is None: continue
        A1 = A0.copy(); A1[2, 1] += bump
        if rho(A1) > 0.82: continue
        Xp, Yp = samp(A0, n0, rng); M0 = ols(Xp, Yp, lam)
        Xq, Yq = samp(A1, n1, rng); refit = ols(Xq, Yq)
        d_en = {h: np.linalg.norm(P(A1, h) - P(A0, h)) for h in horizons}
        scale = max(np.linalg.norm(P(A1, h)) for h in horizons)
        if min(d_en.values()) < 1e-6 * scale: continue
        used += 1
        for mu in mus:
            prox = proximal(M0, Xq, Yq, mu)
            for h in horizons:
                d_e = P(A1, h) - P(A0, h)
                ep_ref = rel(P(refit, h), P(A1, h))
                mm_ref = np.linalg.norm((P(refit, h) - P(M0, h)) - d_e) / d_en[h]
                ep_prox = rel(P(prox, h), P(A1, h))
                mm_prox = np.linalg.norm((P(prox, h) - P(M0, h)) - d_e) / d_en[h]
                cell = acc[mu][h]
                cell["refit_ep"].append(ep_ref); cell["refit_mm"].append(mm_ref)
                cell["prox_ep"].append(ep_prox); cell["prox_mm"].append(mm_prox)
                cell["reverse"].append(ep_ref < ep_prox and mm_prox < mm_ref)
    out = {"used": used, "mus": list(mus), "horizons": list(horizons)}
    for mu in mus:
        out[f"mu{mu:g}"] = {}
        for h in horizons:
            c = acc[mu][h]
            k_reverse = int(np.sum(c["reverse"]))
            lo, hi = wilson_ci(k_reverse, len(c["reverse"]))
            out[f"mu{mu:g}"][f"h{h}"] = {
                "refit_endpoint": float(np.median(c["refit_ep"])),
                "refit_mismatch": float(np.median(c["refit_mm"])),
                "proximal_endpoint": float(np.median(c["prox_ep"])),
                "proximal_mismatch": float(np.median(c["prox_mm"])),
                "rank_disagreement": k_reverse / len(c["reverse"]),
                "disagreement_count": k_reverse,
                "disagreement_ci95": [lo, hi],
            }
    return out

# =========================================================== E10 ==============
def E10(seeds=600, horizons=(1, 5, 12), mu=80., lam=40.,
        n0=400, n1=400, bump=0.35):
    """Angle between inherited error and update mismatch.

    The central identity is vector-valued.  This diagnostic records whether
    the two signed terms reinforce (cosine +1) or cancel (cosine -1).
    """
    rules = ("refit", "proximal", "adapter")
    acc = {r: {h: {"cosine": [], "endpoint_to_triangle": [],
                   "mismatch_to_inherited": []} for h in horizons}
           for r in rules}
    used = 0
    for s in range(seeds):
        rng = np.random.default_rng(s)
        A0 = base_world(rng)
        if A0 is None: continue
        A1 = A0.copy(); A1[2, 1] += bump
        if rho(A1) > 0.82: continue
        Xp, Yp = samp(A0, n0, rng); M0 = ols(Xp, Yp, lam)
        Xq, Yq = samp(A1, n1, rng)
        cand = {
            "refit": ols(Xq, Yq),
            "proximal": proximal(M0, Xq, Yq, mu),
            "adapter": oracle_entry(M0, Xq, Yq, (2, 1)),
        }
        d_en = {h: np.linalg.norm(P(A1, h) - P(A0, h)) for h in horizons}
        scale = max(np.linalg.norm(P(A1, h)) for h in horizons)
        if min(d_en.values()) < 1e-6 * scale: continue
        used += 1
        for h in horizons:
            eps0 = P(M0, h) - P(A0, h)
            d_e = P(A1, h) - P(A0, h)
            n0_err = np.linalg.norm(eps0)
            for name, M1 in cand.items():
                mismatch = (P(M1, h) - P(M0, h)) - d_e
                eps1 = P(M1, h) - P(A1, h)
                nm = np.linalg.norm(mismatch)
                if n0_err <= 1e-12 or nm <= 1e-12: continue
                a = acc[name][h]
                a["cosine"].append(float(np.vdot(eps0, mismatch).real /
                                         (n0_err * nm)))
                a["endpoint_to_triangle"].append(
                    float(np.linalg.norm(eps1) / (n0_err + nm)))
                a["mismatch_to_inherited"].append(float(nm / n0_err))
    out = {"used": used, "mu": mu, "horizons": list(horizons)}
    for name in rules:
        out[name] = {}
        for h in horizons:
            out[name][f"h{h}"] = {
                key: float(np.median(values))
                for key, values in acc[name][h].items()
            }
    return out

def figure4(e6, path="results.json"):
    """Reproduce the base-error-matched linear/nonlinear comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = json.load(open(path))
    linear = np.array(main["A_poor"]["disagreement"][:e6["H"]]) * 100
    nonlinear = np.array(e6["disagreement"]) * 100
    hs = np.arange(1, e6["H"] + 1)
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False,
        "lines.linewidth": 1.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(hs, linear, color="#0072B2", marker="o", ls="-", ms=3.5,
            label="linear (base error 0.100)")
    ax.plot(hs, nonlinear, color="#D55E00", marker="s", ls="--", ms=3.5,
            label=f"nonlinear (base error {e6['inherited_h1']:.3f})")
    ax.set_xlabel("prediction horizon $h$")
    ax.set_ylabel("rank disagreement (% of seeds)")
    ax.set_xticks([1, 2, 4, 6, 8]); ax.set_ylim(-4, 104)
    ax.set_title("Rank reversal survives nonlinear rollout", fontsize=8)
    ax.legend(fontsize=7)
    fig.savefig("fig4_nonlinear.pdf")
    fig.savefig("fig4_nonlinear.png")
    plt.close(fig)

def figure5(e9):
    """Reproduce the non-oracle proximal-strength ablation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False,
        "lines.linewidth": 1.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    mus = np.array(e9["mus"])
    styles = {1: ("#0072B2", "o", "-"),
              5: ("#D55E00", "s", "--"),
              12: ("#009E73", "^", "-.")}
    for h, (color, marker, ls) in styles.items():
        ys = [e9[f"mu{mu:g}"][f"h{h}"]["rank_disagreement"] * 100
              for mu in mus]
        ax.plot(mus, ys, color=color, marker=marker, ls=ls, ms=3.5,
                label=f"$h={h}$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(mus); ax.set_xticklabels([f"{mu:g}" for mu in mus])
    ax.set_xlabel("proximal anchoring strength $\\mu$")
    ax.set_ylabel("rank disagreement (% of seeds)")
    ax.set_ylim(-4, 104); ax.legend(fontsize=7, title="horizon", title_fontsize=7)
    ax.set_title("Rank reversal does not require an oracle update", fontsize=8)
    fig.savefig("fig5_proximal_ablation.pdf")
    fig.savefig("fig5_proximal_ablation.png")
    plt.close(fig)

# ============================================================ main ============
if __name__ == "__main__":
    L = []
    def say(x): print(x); L.append(x)

    say("="*94); say("E5  K-ROUND ACCUMULATION  (limited data per round; one entry changes per round)")
    e5 = E5()
    say(f"  {e5['used']} usable seeds, K={e5['K']} rounds, n_round={e5['n_round']}")
    for h in (1, 5):
        say(f"\n  --- horizon h={h} ---")
        say(f"  {'rule':10s} {'round-1 ENDPT':>14s} {'round-1 MISMCH':>15s} {'mean MISMCH':>13s} {'FINAL ENDPT (K)':>17s}")
        for r in e5["rules"]:
            say(f"  {r:10s} {e5[f'round1_endpoint_h{h}'][r]:14.3f} {e5[f'mismatch_h{h}'][r]:15.3f} "
                f"{e5[f'mismatch_mean_h{h}'][r]:13.3f} {e5[f'final_endpoint_h{h}'][r]:17.3f}")
        r1 = e5[f'round1_endpoint_h{h}']; mm = e5[f'mismatch_h{h}']; fin = e5[f'final_endpoint_h{h}']
        ordr = lambda dd: [k for k, _ in sorted(dd.items(), key=lambda kv: kv[1])]
        say(f"  ranking by round-1 endpoint : {ordr(r1)}")
        say(f"  ranking by round-1 mismatch : {ordr(mm)}")
        say(f"  ranking by FINAL endpoint   : {ordr(fin)}")
        say(f"  round-1 endpoint predicts final? {'YES' if ordr(r1)==ordr(fin) else 'NO'}"
            f"   round-1 mismatch predicts final? {'YES' if ordr(mm)==ordr(fin) else 'NO'}")
        say("  Per-seed agreement with the final ordering:")
        for metric, diag in e5[f"agreement_with_final_h{h}"].items():
            say(f"    {metric:17s} pairwise={diag['pairwise_order_accuracy']:.3f}  "
                f"best={diag['best_rule_accuracy']:.3f}  "
                f"mean Spearman={diag['mean_spearman']:.3f}")
        adv = e5[f"mismatch_advantage_h{h}"]
        say(f"    mismatch - endpoint: pairwise {adv['pairwise_accuracy_difference']:+.3f} "
            f"95% CI [{adv['pairwise_difference_ci95'][0]:+.3f}, "
            f"{adv['pairwise_difference_ci95'][1]:+.3f}]; Spearman "
            f"{adv['mean_spearman_difference']:+.3f} "
            f"[{adv['spearman_difference_ci95'][0]:+.3f}, "
            f"{adv['spearman_difference_ci95'][1]:+.3f}]")

    say(""); say("="*94); say("E6  NONLINEAR STRESS TEST  (x_{t+1} = A tanh(x_t); rollout is nonlinear)")
    e6 = E6()
    say(f"  {e6['used']}/{e6['attempted']} usable, median inherited error h=1 = {e6['inherited_h1']:.3f}")
    say(f"  {'rule':10s}" + "".join(f"{'ENDPT h='+str(h):>12s}" for h in (1, 4, 8))
        + "  |" + "".join(f"{'MISMCH h='+str(h):>12s}" for h in (1, 4, 8)))
    for k in ("refit", "replay", "adapter"):
        ep = e6["median"][k]["endpoint"]; mm = e6["median"][k]["mismatch"]
        say(f"  {k:10s}" + "".join(f"{ep[h-1]:12.3f}" for h in (1, 4, 8))
            + "  |" + "".join(f"{mm[h-1]:12.3f}" for h in (1, 4, 8)))
    say("  rank disagreement: " + "".join(f"  h={h} {e6['disagreement'][h-1]*100:5.1f}%" for h in (1, 2, 4, 8)))

    say(""); say("="*94); say("E7  AGGREGATION  (worst-case column vs global Frobenius error)")
    e7 = E7()
    say(f"  {e7['used']} usable seeds")
    say(f"  {'rule':10s} {'GLOBAL h=1':>11s} {'WORST h=1':>10s} {'ratio':>7s} "
        f"{'GLOBAL h=12':>12s} {'WORST h=12':>11s}")
    for k in ("refit", "replay", "adapter"):
        say(f"  {k:10s} {e7[k]['global'][0]:11.3f} {e7[k]['worst'][0]:10.3f} "
            f"{e7[k]['ratio_h1']:7.2f} {e7[k]['global'][11]:12.3f} {e7[k]['worst'][11]:11.3f}")

    say(""); say("="*94); say("E8  WILSON 95% CIs ON HEADLINE DISAGREEMENT RATES")
    e8 = E8()
    for cond, lbl in (("A_poor", "inherited-bias base"), ("A_good", "low-error base")):
        for h in ("h1", "h5", "h12"):
            v = e8[cond][h]
            say(f"  {lbl:22s} {h:4s}  {v['p']*100:6.1f}%   95% CI [{v['ci95'][0]*100:5.1f}, {v['ci95'][1]*100:5.1f}]  n={v['n']}")

    say(""); say("="*94); say("E9  NON-ORACLE PROXIMAL UPDATE ABLATION")
    e9 = E9()
    say(f"  {e9['used']} usable seeds")
    say(f"  {'mu':>6s}" + "".join(f"{'disagree h='+str(h):>16s}" for h in e9["horizons"]))
    for mu in e9["mus"]:
        say(f"  {mu:6.0f}" + "".join(
            f"{e9[f'mu{mu:g}'][f'h{h}']['rank_disagreement']*100:15.1f}%"
            for h in e9["horizons"]))

    say(""); say("="*94); say("E10  CANCELLATION GEOMETRY  (signed identity; proximal mu=80)")
    e10 = E10()
    say(f"  {e10['used']} usable seeds")
    say(f"  {'rule':10s}" + "".join(f"{'cos h='+str(h):>11s}" for h in e10["horizons"])
        + "  |" + "".join(f"{'endpoint/tri h='+str(h):>18s}" for h in e10["horizons"]))
    for name in ("refit", "proximal", "adapter"):
        say(f"  {name:10s}" + "".join(
            f"{e10[name][f'h{h}']['cosine']:11.3f}" for h in e10["horizons"])
            + "  |" + "".join(
            f"{e10[name][f'h{h}']['endpoint_to_triangle']:18.3f}"
            for h in e10["horizons"]))

    json.dump({"E5": e5, "E6": e6, "E7": e7, "E8": e8, "E9": e9,
               "E10": e10},
              open("results_followup.json", "w"), indent=2)
    open("results_followup.md", "w").write("# Follow-up experiments\n\n```\n" + "\n".join(L) + "\n```\n")
    figure4(e6); figure5(e9)
    print("\nwrote results_followup.md, results_followup.json, "
          "fig4_nonlinear.pdf/.png, fig5_proximal_ablation.pdf/.png")
