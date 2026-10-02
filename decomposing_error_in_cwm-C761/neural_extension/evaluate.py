"""Evaluation: decomposition metrics, exclusions, rank disagreement, figures.

All quantities are computed from SIGNED stacked prediction vectors; norms are
taken only here, at reporting time.  Per horizon h, panel predictions are
stacked into one vector (n_panel * 2 dims) before taking norms, mirroring the
paper's Frobenius-over-interventions convention:

  ENDPOINT(h)  = ||P1_hat - P1_E|| / ||P1_E||
  INHERITED(h) = ||P0_hat - P0_E|| / ||P0_E||
  MISMATCH(h)  = ||(P1_hat - P0_hat) - (P1_E - P0_E)|| / ||P1_E - P0_E||

The identity eps1 = eps0 + (deltaM - deltaE) is asserted pointwise for every
retained (seed, rule, horizon) as a runtime check.

This module is used as a LIBRARY: evaluate_corrected.py imports
compute_seed_metrics, apply_exclusions and wilson from here.  Its own __main__
belongs to a superseded independent-condition run and is not part of the
reported pipeline -- run evaluate_corrected.py instead.
"""

import csv
import os
import pickle
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from simulator import env_panel_predictions  # noqa: E402

# Fixed palette (validated CVD-safe): categorical hues for refit/replay,
# a single-hue sequential ramp for the proximal-mu dose (magnitude job).
RULE_COLORS = {
    "refit": "#0072B2", "replay": "#E69F00",
    "proximal-0.0001": "#A1D99B", "proximal-0.001": "#74C476",
    "proximal-0.01": "#41AB5D", "proximal-0.1": "#238B45",
    "proximal-1": "#005A32",
}
HORIZON_COLORS = ["#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#084594"]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def stack(P):  # (n_horizons, n_panel, 2) -> (n_horizons, n_panel*2)
    return P.reshape(P.shape[0], -1).astype(np.float64)


def compute_seed_metrics(rec, P0E, P1E, rules, horizons):
    """Per-seed metrics from signed vectors + pointwise identity check."""
    P0 = stack(rec["P0hat"])
    dE = P1E - P0E
    eps0 = P0 - P0E
    out = {"inherited": np.linalg.norm(eps0, axis=1) / np.linalg.norm(P0E, axis=1),
           "endpoint": {}, "mismatch": {}, "cos_h1": {}}
    for rule in rules:
        P1 = stack(rec["P1hat"][rule])
        eps1 = P1 - P1E
        dM = P1 - P0
        dMdE = dM - dE
        resid = np.max(np.abs(eps1 - (eps0 + dMdE)))
        scale = max(1.0, float(np.max(np.abs(eps1))))
        assert resid <= 1e-9 * scale, f"identity violated: {resid} (scale {scale})"
        out["endpoint"][rule] = np.linalg.norm(eps1, axis=1) / np.linalg.norm(P1E, axis=1)
        out["mismatch"][rule] = np.linalg.norm(dMdE, axis=1) / np.linalg.norm(dE, axis=1)
        e0, d0 = eps0[0], dMdE[0]  # horizons[0] == 1 by config
        out["cos_h1"][rule] = float(
            e0 @ d0 / (np.linalg.norm(e0) * np.linalg.norm(d0) + 1e-300))
    return out


def apply_exclusions(cfg, records):
    """Predeclared exclusions; returns (retained, accounting dict per n0)."""
    factor = cfg["exclusions"]["base_mse_median_factor"]
    by_n0 = {}
    for r in records:
        by_n0.setdefault(r["n0"], []).append(r)
    retained, accounting = [], {}
    for n0, recs in sorted(by_n0.items()):
        med = float(np.median([r["base_val_mse"] for r in recs]))
        keep, n_mse, n_div = [], 0, 0
        for r in recs:
            bad_mse = r["base_val_mse"] > factor * med
            if bad_mse:
                n_mse += 1
            if r["diverged"]:
                n_div += 1
            if not (bad_mse or r["diverged"]):
                keep.append(r)
        accounting[n0] = {"attempted": len(recs), "retained": len(keep),
                          "excluded_mse": n_mse, "excluded_diverged": n_div,
                          "median_base_mse": med}
        retained.extend(keep)
    return retained, accounting


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(results_path):
    with open(results_path, "rb") as f:
        blob = pickle.load(f)
    cfg, records = blob["cfg"], blob["records"]
    horizons = cfg["horizons"]
    rules = (["refit", "replay"]
             + [f"proximal-{mu:g}" for mu in cfg["update_rules"]["proximal"]["mus"]])

    envH = np.array(horizons)
    P0E = stack(env_panel_predictions(cfg, "v0")[envH])
    P1E = stack(env_panel_predictions(cfg, "v1")[envH])

    # Zero-effect accounting (per-query ||deltaE||, predeclared threshold).
    P0E_pq = env_panel_predictions(cfg, "v0")[envH]  # (H, n_panel, 2)
    P1E_pq = env_panel_predictions(cfg, "v1")[envH]
    dE_pq = np.linalg.norm(P1E_pq - P0E_pq, axis=2)  # (H, n_panel)
    thr = cfg["zero_effect_threshold"]
    zero_rows = [(h, int((dE_pq[i] < thr).sum()), float(dE_pq[i].min()),
                  float(np.linalg.norm(P1E[i] - P0E[i])))
                 for i, h in enumerate(horizons)]

    retained, accounting = apply_exclusions(cfg, records)
    print("Seed accounting (attempted -> retained):")
    for n0, a in sorted(accounting.items()):
        print(f"  N0={n0:>4}: {a['attempted']} -> {a['retained']} "
              f"(mse fail {a['excluded_mse']}, diverged {a['excluded_diverged']})")

    # Per-seed metrics.
    per_n0 = {}
    for r in retained:
        m = compute_seed_metrics(r, P0E, P1E, rules, horizons)
        per_n0.setdefault(r["n0"], []).append(m)

    n0_grid = sorted(per_n0.keys())
    n_h = len(horizons)

    # ---- results table: median metrics per (n0, rule, horizon) ----
    med = {}   # (n0, rule) -> dict of arrays
    inh_med = {}
    rows = []
    for n0 in n0_grid:
        ms = per_n0[n0]
        inh = np.median(np.stack([m["inherited"] for m in ms]), axis=0)
        inh_med[n0] = inh
        for rule in rules:
            ep = np.median(np.stack([m["endpoint"][rule] for m in ms]), axis=0)
            mm = np.median(np.stack([m["mismatch"][rule] for m in ms]), axis=0)
            med[(n0, rule)] = {"endpoint": ep, "mismatch": mm}
            for i, h in enumerate(horizons):
                rows.append([n0, rule, h, f"{ep[i]:.4f}", f"{inh[i]:.4f}",
                             f"{mm[i]:.4f}", accounting[n0]["attempted"],
                             accounting[n0]["retained"]])
    write_csv(os.path.join(BASE, "results", "results_table.csv"),
              ["n0", "rule", "horizon", "median_endpoint", "median_inherited",
               "median_mismatch", "seeds_attempted", "seeds_retained"], rows)

    # ---- rank disagreement per (n0, horizon) ----
    dis_rows = []
    disagreement = {}
    for n0 in n0_grid:
        ms = per_n0[n0]
        n = len(ms)
        i_refit = rules.index("refit")
        i_prox = rules.index("proximal-1")
        for i, h in enumerate(horizons):
            binary = 0
            rp_flips = 0  # refit vs most-anchored proximal: the closest
            pw = []       # analogue of the paper's refit-vs-oracle statistic
            for m in ms:
                e = np.array([m["endpoint"][r][i] for r in rules])
                q = np.array([m["mismatch"][r][i] for r in rules])
                binary += int(np.argmin(e) != np.argmin(q))
                rp_flips += int((e[i_refit] < e[i_prox]) != (q[i_refit] < q[i_prox]))
                flips = sum(1 for a in range(len(rules)) for b in range(a + 1, len(rules))
                            if (e[a] < e[b]) != (q[a] < q[b]))
                pw.append(flips / (len(rules) * (len(rules) - 1) / 2))
            lo, hi = wilson(binary, n)
            rp_lo, rp_hi = wilson(rp_flips, n)
            disagreement[(n0, h)] = {"rate": binary / n, "lo": lo, "hi": hi,
                                     "k": binary, "n": n, "pairwise": float(np.mean(pw)),
                                     "rp": rp_flips / n, "rp_lo": rp_lo, "rp_hi": rp_hi}
            dis_rows.append([n0, h, n, binary, f"{binary / n:.4f}",
                             f"{lo:.4f}", f"{hi:.4f}", f"{np.mean(pw):.4f}",
                             rp_flips, f"{rp_lo:.4f}", f"{rp_hi:.4f}"])
    write_csv(os.path.join(BASE, "results", "disagreement.csv"),
              ["n0", "horizon", "n_retained", "reversals_binary", "rate",
               "wilson_lo", "wilson_hi", "mean_pairwise_disagreement",
               "refit_vs_proximal1_flips", "rp_wilson_lo", "rp_wilson_hi"], dis_rows)

    # ---- cosine geometry at h = 1 ----
    cos_rows = []
    cos_med = {}
    for n0 in n0_grid:
        for rule in rules:
            c = np.array([m["cos_h1"][rule] for m in per_n0[n0]])
            cos_med[(n0, rule)] = float(np.median(c))
            cos_rows.append([n0, rule, f"{np.median(c):.4f}",
                             f"{np.percentile(c, 25):.4f}", f"{np.percentile(c, 75):.4f}"])
    write_csv(os.path.join(BASE, "results", "cosines.csv"),
              ["n0", "rule", "median_cos_h1", "q25", "q75"], cos_rows)

    write_csv(os.path.join(BASE, "results", "zero_effect.csv"),
              ["horizon", "n_queries_below_threshold", "min_per_query_deltaE_norm",
               "global_deltaE_norm"], zero_rows)

    render_markdown(cfg, accounting, med, inh_med, disagreement, cos_med,
                    rules, horizons, n0_grid, zero_rows)
    make_figures(cfg, per_n0, med, inh_med, disagreement, cos_med,
                 rules, horizons, n0_grid)

    # Console headline.
    hn0 = cfg["data"]["headline_n0"]
    print(f"\nHeadline (N0={hn0}):")
    for i, h in enumerate(horizons):
        d = disagreement[(hn0, h)]
        print(f"  h={h:>2}: best-by-endpoint != best-by-mismatch on "
              f"{d['k']}/{d['n']} seeds = {100 * d['rate']:.1f}% "
              f"(Wilson [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}])")
    print("\nMedian cos(eps0, deltaM - deltaE) at h=1, headline N0:")
    for rule in rules:
        print(f"  {rule:>16}: {cos_med[(hn0, rule)]:+.3f}")


def render_markdown(cfg, accounting, med, inh_med, disagreement, cos_med,
                    rules, horizons, n0_grid, zero_rows):
    lines = ["# Results tables", ""]
    lines += ["## Seed accounting", "",
              "| N0 | attempted | retained | excluded (MSE) | excluded (diverged) |",
              "|---:|---:|---:|---:|---:|"]
    for n0 in n0_grid:
        a = accounting[n0]
        lines.append(f"| {n0} | {a['attempted']} | {a['retained']} | "
                     f"{a['excluded_mse']} | {a['excluded_diverged']} |")
    lines += ["", "## Zero-effect queries (threshold "
              f"{cfg['zero_effect_threshold']:g})", "",
              "| horizon | queries below threshold | min per-query ||dE|| | global ||dE|| |",
              "|---:|---:|---:|---:|"]
    for h, c, mn, gl in zero_rows:
        lines.append(f"| {h} | {c} | {mn:.4g} | {gl:.4g} |")
    for n0 in n0_grid:
        lines += ["", f"## N0 = {n0}: median relative errors "
                  "(ENDPOINT / INHERITED / MISMATCH)", "",
                  "| rule | " + " | ".join(f"h={h}" for h in horizons) + " |",
                  "|---|" + "---|" * len(horizons)]
        inh = inh_med[n0]
        lines.append("| (inherited, base model) | "
                     + " | ".join(f"{inh[i]:.3f}" for i in range(len(horizons))) + " |")
        for rule in rules:
            ep, mm = med[(n0, rule)]["endpoint"], med[(n0, rule)]["mismatch"]
            lines.append(f"| {rule} | " + " | ".join(
                f"{ep[i]:.3f} / {mm[i]:.3f}" for i in range(len(horizons))) + " |")
        lines += ["", f"### N0 = {n0}: rank disagreement "
                  "(best-by-endpoint != best-by-mismatch; mean pairwise flips)", "",
                  "| horizon | rate | 95% Wilson | mean pairwise |",
                  "|---:|---:|---|---:|"]
        for h in horizons:
            d = disagreement[(n0, h)]
            lines.append(f"| {h} | {d['k']}/{d['n']} = {100 * d['rate']:.1f}% | "
                         f"[{100 * d['lo']:.1f}, {100 * d['hi']:.1f}] | "
                         f"{100 * d['pairwise']:.1f}% |")
    lines += ["", "## Median cos(eps0, deltaM - deltaE) at h = 1", "",
              "| N0 | " + " | ".join(rules) + " |",
              "|---:|" + "---:|" * len(rules)]
    for n0 in n0_grid:
        lines.append(f"| {n0} | " + " | ".join(
            f"{cos_med[(n0, r)]:+.3f}" for r in rules) + " |")
    with open(os.path.join(BASE, "results", "results_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def make_figures(cfg, per_n0, med, inh_med, disagreement, cos_med,
                 rules, horizons, n0_grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    hn0 = cfg["data"]["headline_n0"]
    rule_label = {r: r for r in rules}
    rule_label["refit"] = "refit (discard base)"
    rule_label["replay"] = "replay (1:1 old+new)"

    # Figure 1: median ENDPOINT and MISMATCH vs horizon.  Main figure at the
    # predeclared headline N0; supplementary version at N0 = min(grid), the
    # high-inherited-error regime (closest analogue of the paper's Fig. 1
    # "inherited-bias base" condition).
    def fig1(n0, fname, tag):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
        for ax, key, title in zip(
                axes, ["endpoint", "mismatch"],
                ["Endpoint error (post-update fidelity)",
                 "Update mismatch  $\\|\\delta_M-\\delta_E\\|$ (relative)"]):
            for rule in rules:
                style = "--" if rule.startswith("proximal") else "-"
                ax.plot(horizons, med[(n0, rule)][key], style, marker="o", ms=3.5,
                        lw=1.6, color=RULE_COLORS[rule], label=rule_label[rule])
            ax.set_xlabel("prediction horizon $h$")
            ax.set_ylabel("relative error")
            ax.set_title(title, fontsize=9.5)
            ax.set_xticks(horizons)
            ax.grid(alpha=0.25, lw=0.5)
        axes[0].plot(horizons, inh_med[n0], ":", color="#555555", lw=1.4,
                     label="inherited (base $M^0$)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                   fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Neural world model, $N_0$ = {n0} ({tag}): "
                     "the two views of the same updates", fontsize=10)
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        fig.savefig(os.path.join(BASE, "figures", fname), bbox_inches="tight")
        plt.close(fig)

    fig1(hn0, "fig1_endpoint_mismatch.png", "predeclared headline")
    fig1(min(n0_grid), "fig1b_endpoint_mismatch_n0min.png",
         "high inherited error")

    # Figure 2: disagreement vs measured inherited error (N0 sweep), per
    # horizon.  Left: headline binary event; right: mean pairwise flips.
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharex=True)
    xs0 = np.array([inh_med[n0][0] for n0 in n0_grid])  # measured one-step inherited
    order = np.argsort(xs0)
    for i, h in enumerate(horizons):
        ys = np.array([100 * disagreement[(n0, h)]["rate"] for n0 in n0_grid])[order]
        los = np.array([100 * disagreement[(n0, h)]["lo"] for n0 in n0_grid])[order]
        his = np.array([100 * disagreement[(n0, h)]["hi"] for n0 in n0_grid])[order]
        pw = np.array([100 * disagreement[(n0, h)]["pairwise"] for n0 in n0_grid])[order]
        axes[0].plot(xs0[order], ys, marker="o", ms=4, lw=1.6,
                     color=HORIZON_COLORS[i], label=f"$h$ = {h}")
        axes[0].fill_between(xs0[order], los, his, color=HORIZON_COLORS[i],
                             alpha=0.12, lw=0)
        axes[1].plot(xs0[order], pw, marker="o", ms=4, lw=1.6,
                     color=HORIZON_COLORS[i], label=f"$h$ = {h}")
    axes[0].set_title("best-by-endpoint $\\neq$ best-by-mismatch", fontsize=9.5)
    axes[1].set_title("mean pairwise rank flips (21 rule pairs)", fontsize=9.5)
    for ax in axes:
        ax.set_xlabel("measured inherited error  $\\|\\varepsilon^0\\|/\\|P^0_E\\|$ at $h=1$")
        ax.set_ylabel("rank disagreement (% )")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.25, lw=0.5)
    axes[1].legend(frameon=False, fontsize=8, title="horizon", title_fontsize=8)
    fig.suptitle("Disagreement grows with inherited error ($N_0$ sweep)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(BASE, "figures", "fig2_disagreement_vs_inherited.png"),
                bbox_inches="tight")
    plt.close(fig)

    # Figure 3: cosine geometry at h = 1, headline N0 (small-multiple histograms).
    ms = per_n0[hn0]
    fig, axes = plt.subplots(1, len(rules), figsize=(2.0 * len(rules), 2.3),
                             sharex=True, sharey=True)
    bins = np.linspace(-1, 1, 41)
    for ax, rule in zip(axes, rules):
        c = np.array([m["cos_h1"][rule] for m in ms])
        ax.hist(c, bins=bins, color=RULE_COLORS[rule], edgecolor="white", lw=0.3)
        ax.axvline(np.median(c), color="#333333", lw=1.0, ls="--")
        ax.set_title(f"{rule}\nmed {np.median(c):+.2f}", fontsize=8)
        ax.set_xlabel("cos", fontsize=8)
        ax.grid(alpha=0.2, lw=0.4)
    axes[0].set_ylabel("seeds")
    fig.suptitle("cos$(\\varepsilon^0,\\ \\delta_M-\\delta_E)$ at $h=1$, "
                 f"$N_0$ = {hn0} (−1 = update cancels inherited error)", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(BASE, "figures", "fig3_cosines.png"), bbox_inches="tight")
    plt.close(fig)
    print("Figures written to figures/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.join(BASE, "results", "results.pkl"))
