"""
Grading the Update, Not the Model -- Decomposing Error in Continual World Models
Constructed demonstration on a linear dynamical SCM.  Single source of truth for
every number and figure in the paper.

    python3 delta_contract_experiment.py            # tables + figures + results.md
    python3 delta_contract_experiment.py --nofig    # tables only

------------------------------------------------------------------------------
SETUP
    x_{t+1} = A x_t + noise,  A in R^{d x d}.  The world changes ONE entry of A
    (a single mechanism shift):  A1 = A0 with A0[CH] += BUMP.

    Intervention family:  do(x_0 = e_i), unit i, prediction horizon h.
    Chosen because it makes the environment delta available in CLOSED FORM,
    so the demonstration has a genuine interventional oracle:

        delta_E(h) = A1^h - A0^h                 (environment change)
        delta_M(h) = A1h^h - A0h^h               (model change)
        eps^v(h)   = Av_hat^h - Av^h             (model error at version v)

    IDENTITY (verified numerically in report_identity):
        delta_M - delta_E == eps^1 - eps^0

    Three reported quantities, all relative Frobenius (see AGGREGATION note):
        ENDPOINT(h) = ||A1h^h - A1^h|| / ||A1^h||          post-update fidelity
        BASE(h)     = ||A0h^h - A0^h|| / ||A0^h||          inherited error
        MISMATCH(h) = ||delta_M - delta_E|| / ||delta_E||  update mismatch

------------------------------------------------------------------------------
AGGREGATION NOTE (honest description, per protocol item 6)
    All three quantities aggregate over the intervention family by Frobenius
    norm, i.e. a ratio of root-sum-square errors over unit interventions.  This
    is a global, mean-like aggregation.  E7 tests a coherent worst-column norm
    and finds no discriminative advantage in this construction.  A naive
    per-query relative supremum is undefined when the true delta is zero and
    must not silently drop those queries.  The protocol therefore requires the
    aggregation and normalization to be declared; it does not endorse a
    universal worst-case scalar.

ORACLE NOTE (per protocol item 7)
    `adapter` is an ORACLE DIAGNOSTIC, not a proposed algorithm.  It is told
    exactly which entry changed.  It exists to instantiate "a faithful update
    applied to an inherited-bias model", which is the regime where endpoint
    fidelity and update mismatch disagree.  It is not a method, is not
    deployable, and must never be reported as a baseline that "wins".
------------------------------------------------------------------------------
"""
import json, sys
import numpy as np

# ----------------------------------------------------------------- config ----
D, H, CH, BUMP = 5, 12, (2, 1), 0.35
SEEDS          = 600
SIGMA          = 0.25
N_POST         = 400
RHO_TARGET     = 0.70
RHO_SLACK      = 0.12
RULES          = ("refit", "replay", "adapter")
HORIZONS_SHOWN = (1, 2, 3, 5, 8, 12)

P   = lambda M, h: np.linalg.matrix_power(M, h)
rho = lambda M: abs(np.linalg.eigvals(M)).max()
rel = lambda X, Y: np.linalg.norm(X - Y) / np.linalg.norm(Y)

# ------------------------------------------------------------- primitives ----
def world(rng, target_rho=RHO_TARGET, bump=BUMP):
    """Returns (A0, A1) or a string naming the rejection reason."""
    A0 = rng.normal(0, .45, (D, D)) * (rng.random((D, D)) < .55)
    r = rho(A0)
    if r < 1e-6:
        return "degenerate_A0"
    A0 *= target_rho / r
    A1 = A0.copy(); A1[CH] += bump
    if rho(A1) > target_rho + RHO_SLACK:
        return "rho_A1_too_large"
    return A0, A1

def samp(A, n, sigma, rng):
    X = rng.normal(0, 1, (D, n))
    return X, A @ X + rng.normal(0, sigma, (D, n))

def ols(X, Y, lam=0.):
    return Y @ X.T @ np.linalg.inv(X @ X.T + lam * np.eye(D))

def adapter(A0h, X, Y):
    """ORACLE DIAGNOSTIC -- only the entry that actually changed is free."""
    r, c = CH
    A = A0h.copy()
    resid = Y[r] - A0h[r] @ X + A0h[r, c] * X[c]
    A[r, c] = resid @ X[c] / (X[c] @ X[c])
    return A

# ---------------------------------------------------------------- harness ----
def run_condition(lam, n0, seeds=SEEDS, target_rho=RHO_TARGET):
    """One (base-quality, dynamics) cell.  Returns per-seed curves + accounting."""
    out = {r: {"endpoint": [], "mismatch": []} for r in RULES}
    base_curves, reject = [], {"degenerate_A0": 0, "rho_A1_too_large": 0,
                               "change_does_not_propagate": 0, "nonfinite": 0}
    attempted = usable = 0

    for s in range(seeds):
        attempted += 1
        rng = np.random.default_rng(s)
        w = world(rng, target_rho)
        if isinstance(w, str):
            reject[w] += 1; continue
        A0, A1 = w

        Xp, Yp = samp(A0, n0, SIGMA, rng)
        A0h = ols(Xp, Yp, lam)                       # the BASE model
        Xq, Yq = samp(A1, N_POST, SIGMA, rng)        # post-change data

        cand = {
            "refit":   ols(Xq, Yq),                                     # discard base
            # 1:1 mixture -- cap the pre-change half at N_POST so `replay` is
            # comparable across base-quality conditions (n0 varies, the mixture must not).
            "replay":  ols(np.hstack([Xp[:, -N_POST:], Xq]), np.hstack([Yp[:, -N_POST:], Yq])),
            "adapter": adapter(A0h, Xq, Yq),                            # ORACLE diagnostic
        }

        dEn = np.array([np.linalg.norm(P(A1, h) - P(A0, h)) for h in range(1, H + 1)])
        scale = max(np.linalg.norm(P(A1, h)) for h in range(1, H + 1))
        if dEn.min() < 1e-6 * scale:
            reject["change_does_not_propagate"] += 1; continue

        base = np.array([rel(P(A0h, h), P(A0, h)) for h in range(1, H + 1)])
        cell, ok = {}, True
        for k, A1h in cand.items():
            ep = np.array([rel(P(A1h, h), P(A1, h)) for h in range(1, H + 1)])
            mm = np.array([np.linalg.norm((P(A1h, h) - P(A0h, h)) - (P(A1, h) - P(A0, h)))
                           / dEn[h - 1] for h in range(1, H + 1)])
            if not (np.all(np.isfinite(ep)) and np.all(np.isfinite(mm))):
                ok = False
            cell[k] = (ep, mm)
        if not ok or not np.all(np.isfinite(base)):
            reject["nonfinite"] += 1; continue

        usable += 1
        base_curves.append(base)
        for k in RULES:
            out[k]["endpoint"].append(cell[k][0]); out[k]["mismatch"].append(cell[k][1])

    return {"lam": lam, "n0": n0, "target_rho": target_rho,
            "attempted": attempted, "usable": usable, "reject": reject,
            "base": np.array(base_curves),
            "curves": {k: {m: np.array(v) for m, v in out[k].items()} for k in RULES}}

def disagreement(res, a="refit", b="adapter"):
    """Fraction of seeds where endpoint fidelity and update mismatch DISAGREE
    on the ordering of two update rules, per horizon."""
    ea, eb = res["curves"][a]["endpoint"], res["curves"][b]["endpoint"]
    ma, mb = res["curves"][a]["mismatch"], res["curves"][b]["mismatch"]
    return np.mean((ea < eb) & (mb < ma), axis=0)

# ----------------------------------------------------------------- report ----
def report_identity():
    rng = np.random.default_rng(0)
    A0, A1 = world(rng)
    Xp, Yp = samp(A0, N_POST, SIGMA, rng); A0h = ols(Xp, Yp, 40.)
    Xq, Yq = samp(A1, N_POST, SIGMA, rng); A1h = adapter(A0h, Xq, Yq)
    worst = 0.
    for h in range(1, H + 1):
        lhs = (P(A1h, h) - P(A0h, h)) - (P(A1, h) - P(A0, h))
        rhs = (P(A1h, h) - P(A1, h)) - (P(A0h, h) - P(A0, h))
        worst = max(worst, np.abs(lhs - rhs).max())
    return worst

def table(res, label, lines):
    lines.append(f"\n{'='*94}\n{label}")
    lines.append(f"  seeds: {res['attempted']} attempted, {res['usable']} usable "
                 f"({100*res['usable']/res['attempted']:.1f}%)   rejected: "
                 + ", ".join(f"{k}={v}" for k, v in res["reject"].items() if v))
    lines.append(f"  median inherited error BASE(h=1) = {np.median(res['base'][:,0]):.3f}")
    hdr = (f"{'':9s}" + "".join(f"{'ENDPT h='+str(h):>12s}" for h in (1, 5, 12))
           + "  |" + "".join(f"{'MISMCH h='+str(h):>12s}" for h in (1, 5, 12)))
    lines.append(hdr)
    for k in RULES:
        ep = np.median(res["curves"][k]["endpoint"], 0)
        mm = np.median(res["curves"][k]["mismatch"], 0)
        tag = k + (" *" if k == "adapter" else "")
        lines.append(f"{tag:9s}" + "".join(f"{ep[h-1]:12.3f}" for h in (1, 5, 12))
                     + "  |" + "".join(f"{mm[h-1]:12.3f}" for h in (1, 5, 12)))
    lines.append("  * adapter = ORACLE DIAGNOSTIC, not a proposed algorithm")
    d = disagreement(res)
    lines.append("  RANK DISAGREEMENT (refit better on ENDPOINT and adapter better on MISMATCH):")
    lines.append("    " + "".join(f"  h={h:<2d} {d[h-1]*100:5.1f}%" for h in HORIZONS_SHOWN))
    return d

# ------------------------------------------------------------------- main ----
def main(make_figs=True):
    L, blob = [], {}
    L.append("IDENTITY CHECK")
    w = report_identity()
    L.append(f"  max |(delta_M - delta_E) - (eps^1 - eps^0)| over h=1..{H}  =  {w:.2e}")
    blob["identity_residual"] = float(w)

    # --- Experiment A: two base-quality regimes -------------------------------
    poor = run_condition(lam=40., n0=400)
    good = run_condition(lam=0.,  n0=8000)
    d_poor = table(poor, "EXPERIMENT A1 -- INHERITED-BIAS BASE  (ridge lam=40, n=400)", L)
    d_good = table(good, "EXPERIMENT A2 -- LOW-ERROR BASE      (OLS, n=8000)",          L)

    # --- Experiment B: base-error sweep (the headline figure) -----------------
    L.append(f"\n{'='*94}\nEXPERIMENT B -- BASE-ERROR SWEEP  (lam in 0,5,10,20,40,80 at n=400; plus n=8000)")
    sweep = []
    for lam in (0., 5., 10., 20., 40., 80.):
        sweep.append(run_condition(lam=lam, n0=400))
    sweep.append(run_condition(lam=0., n0=8000))
    L.append(f"{'lam':>6s} {'n0':>6s} {'usable':>7s} {'BASE(1)':>9s}"
             + "".join(f"{'dis h='+str(h):>10s}" for h in (1, 3, 5, 12)))
    sweep_rows = []
    for r in sweep:
        d = disagreement(r); b = float(np.median(r["base"][:, 0]))
        L.append(f"{r['lam']:6.0f} {r['n0']:6d} {r['usable']:7d} {b:9.3f}"
                 + "".join(f"{d[h-1]*100:9.1f}%" for h in (1, 3, 5, 12)))
        sweep_rows.append({"lam": r["lam"], "n0": r["n0"], "usable": r["usable"],
                           "base_h1": b, "disagreement": d.tolist()})

    # --- Experiment C: spectral sweep (appendix) ------------------------------
    L.append(f"\n{'='*94}\nEXPERIMENT C -- SPECTRAL SWEEP  (adapter MISMATCH vs horizon, inherited-bias base)")
    L.append(f"{'rho(A0)':>8s} {'usable':>7s}" + "".join(f"{'h='+str(h):>9s}" for h in HORIZONS_SHOWN) + "   growth")
    spec_rows = []
    for tr in (0.70, 0.90, 0.95, 1.02):
        r = run_condition(lam=40., n0=400, target_rho=tr)
        m = np.median(r["curves"]["adapter"]["mismatch"], 0)
        L.append(f"{tr:8.2f} {r['usable']:7d}" + "".join(f"{m[h-1]:9.3f}" for h in HORIZONS_SHOWN)
                 + f"   {m[H-1]/m[0]:6.1f}x")
        spec_rows.append({"rho": tr, "usable": r["usable"], "mismatch": m.tolist(),
                          "growth": float(m[H-1]/m[0])})

    blob.update({
        "config": {"D": D, "H": H, "changed_entry": list(CH), "bump": BUMP,
                   "seeds": SEEDS, "sigma": SIGMA, "n_post": N_POST,
                   "rho_target": RHO_TARGET, "aggregation": "relative Frobenius (RMS over unit interventions)"},
        "A_poor": {"attempted": poor["attempted"], "usable": poor["usable"],
                   "reject": poor["reject"], "base_h1": float(np.median(poor["base"][:, 0])),
                   "disagreement": d_poor.tolist(),
                   "median": {k: {m: np.median(poor["curves"][k][m], 0).tolist()
                                  for m in ("endpoint", "mismatch")} for k in RULES}},
        "A_good": {"attempted": good["attempted"], "usable": good["usable"],
                   "reject": good["reject"], "base_h1": float(np.median(good["base"][:, 0])),
                   "disagreement": d_good.tolist(),
                   "median": {k: {m: np.median(good["curves"][k][m], 0).tolist()
                                  for m in ("endpoint", "mismatch")} for k in RULES}},
        "B_sweep": sweep_rows, "C_spectral": spec_rows,
    })

    txt = "\n".join(L)
    print(txt)
    open("results.md", "w").write("# Experiment results\n\n```\n" + txt + "\n```\n")
    json.dump(blob, open("results.json", "w"), indent=2)
    print("\nwrote results.md, results.json")

    if make_figs:
        figures(poor, good, d_poor, d_good, sweep)

# ---------------------------------------------------------------- figures ----
def figures(poor, good, d_poor, d_good, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    # Okabe-Ito subset, validated: all six checks PASS.
    # Grayscale print + CVD: identity is ALSO carried by marker and linestyle.
    STY = {"refit":   dict(color="#0072B2", marker="o", ls="-",  label="refit (discard base)"),
           "replay":  dict(color="#D55E00", marker="s", ls="--", label="replay (1:1 old+new)"),
           "adapter": dict(color="#009E73", marker="^", ls="-.", label="adapter (oracle diagnostic)")}
    hs = np.arange(1, H + 1)

    # FIGURE 1 -- endpoint and update mismatch vs horizon (inherited-bias base)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for k in RULES:
        ax[0].plot(hs, np.median(poor["curves"][k]["endpoint"], 0), ms=3.5, **STY[k])
        ax[1].plot(hs, np.median(poor["curves"][k]["mismatch"], 0), ms=3.5, **STY[k])
    ax[0].set_title("Endpoint error  (post-update fidelity)", fontsize=8)
    ax[1].set_title("Update mismatch  $\\|\\delta_M-\\delta_E\\|$", fontsize=8)
    for a in ax:
        a.set_xlabel("prediction horizon $h$"); a.set_ylabel("relative error"); a.set_xticks([1,3,5,8,12])
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Inherited-bias base: the two views rank the update rules oppositely",
                 fontsize=8.5, y=1.04)
    fig.savefig("fig1_endpoint_vs_mismatch.pdf"); fig.savefig("fig1_endpoint_vs_mismatch.png"); plt.close(fig)

    # FIGURE 2 -- rank disagreement vs horizon, both base regimes
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(hs, d_poor*100, color="#0072B2", marker="o", ls="-",  ms=3.5, label="inherited-bias base")
    ax.plot(hs, d_good*100, color="#D55E00", marker="s", ls="--", ms=3.5, label="low-error base")
    ax.set_xlabel("prediction horizon $h$"); ax.set_ylabel("rank disagreement (% of seeds)")
    ax.set_ylim(-4, 104); ax.set_xticks([1,3,5,8,12]); ax.legend(fontsize=7)
    ax.set_title("Disagreement is driven by inherited error", fontsize=8)
    fig.savefig("fig2_disagreement_vs_horizon.pdf"); fig.savefig("fig2_disagreement_vs_horizon.png"); plt.close(fig)

    # FIGURE 3 -- rank disagreement vs MEASURED base error (headline)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    xs = [float(np.median(r["base"][:, 0])) for r in sweep]
    o  = np.argsort(xs); xs = np.array(xs)[o]
    hstyle = {1: ("#0072B2","o","-"), 3: ("#D55E00","s","--"), 5: ("#009E73","^","-."), 12: ("#4d4d4d","D",":")}
    for h,(c,m,ls) in hstyle.items():
        ys = np.array([disagreement(r)[h-1]*100 for r in sweep])[o]
        ax.plot(xs, ys, color=c, marker=m, ls=ls, ms=3.5, label=f"$h={h}$")
    ax.set_xlabel("measured inherited error  $\\|\\epsilon^0\\|/\\|A_0\\|$  at $h{=}1$")
    ax.set_ylabel("rank disagreement (% of seeds)")
    ax.set_ylim(-4, 104); ax.legend(fontsize=7, title="horizon", title_fontsize=7)
    ax.set_title("Disagreement grows with inherited error", fontsize=8)
    fig.savefig("fig3_disagreement_vs_base_error.pdf"); fig.savefig("fig3_disagreement_vs_base_error.png"); plt.close(fig)

    print("wrote fig1_endpoint_vs_mismatch.pdf/.png, fig2_disagreement_vs_horizon.pdf/.png, "
          "fig3_disagreement_vs_base_error.pdf/.png")

if __name__ == "__main__":
    main(make_figs="--nofig" not in sys.argv)
