"""Build the three-panel main figure used by paper.tex.

Panels (a,b) reproduce the controlled linear mirror; panel (c) reads only the
corrected paired neural results.  The original experiment outputs are not
modified.
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from delta_contract_experiment import H, RULES, run_condition


plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.bbox": "tight",
    "font.size": 7.2,
    "axes.titlesize": 7.8,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.45,
    "axes.axisbelow": True,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

LINEAR_STYLE = {
    "refit": dict(color="#0072B2", marker="o", ls="-", label="refit"),
    "replay": dict(color="#D55E00", marker="s", ls="--", label="replay"),
    "adapter": dict(color="#009E73", marker="^", ls="-.", label="oracle diagnostic"),
}


def corrected_h1_rows():
    path = "neural_extension/results_corrected/directional.csv"
    with open(path, newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["horizon"]) == 1]
    rows.sort(key=lambda row: float(row["median_inherited"]))
    return rows


def main():
    linear = run_condition(lam=40.0, n0=400)
    horizons = np.arange(1, H + 1)
    rows = corrected_h1_rows()

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.18),
                             gridspec_kw={"width_ratios": [1, 1, 0.95]})

    for rule in RULES:
        axes[0].plot(horizons,
                     np.median(linear["curves"][rule]["endpoint"], axis=0),
                     ms=2.8, **LINEAR_STYLE[rule])
        axes[1].plot(horizons,
                     np.median(linear["curves"][rule]["mismatch"], axis=0),
                     ms=2.8, **LINEAR_STYLE[rule])

    axes[0].set_title("(a) Endpoint error")
    axes[1].set_title("(b) Update mismatch")
    for axis in axes[:2]:
        axis.set_xlabel("prediction horizon $h$")
        axis.set_ylabel("relative error")
        axis.set_xticks([1, 3, 5, 8, 12])

    inherited = np.asarray([float(row["median_inherited"]) for row in rows])
    expected = 100 * np.asarray([float(row["expected_rate"]) for row in rows])
    lower = 100 * np.asarray([float(row["expected_wilson_lo"]) for row in rows])
    upper = 100 * np.asarray([float(row["expected_wilson_hi"]) for row in rows])
    pairwise = 100 * np.asarray([
        float(row["mean_pairwise_disagreement"]) for row in rows])

    axes[2].plot(inherited, expected, color="#0072B2", marker="o", ms=3.0,
                 label="directional pair")
    axes[2].fill_between(inherited, lower, upper, color="#0072B2",
                         alpha=0.14, linewidth=0)
    axes[2].plot(inherited, pairwise, color="#D55E00", marker="s", ms=3.0,
                 label="mean pairwise")
    axes[2].set_title("(c) Neural paired sweep")
    axes[2].set_xlabel("inherited error at $h=1$")
    axes[2].set_ylabel("rank disagreement (%)")
    axes[2].set_ylim(-3, 103)
    axes[2].legend(loc="upper left", frameon=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", ncol=3, frameon=False,
               bbox_to_anchor=(0.045, -0.015), handlelength=2.1,
               columnspacing=0.9)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.26, wspace=0.42)
    fig.savefig("fig1_combined_results.pdf")
    fig.savefig("fig1_combined_results.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
