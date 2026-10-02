"""Evaluation for the corrected paired neural experiment."""

import csv
import os
import pickle
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from evaluate import apply_exclusions, compute_seed_metrics, wilson  # noqa: E402
from simulator import env_panel_predictions  # noqa: E402


def write_csv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def bootstrap_median_ci(values, repetitions, rng):
    values = np.asarray(values, dtype=np.float64)
    sample_ids = rng.integers(0, len(values), size=(repetitions, len(values)))
    boot = np.median(values[sample_ids], axis=1)
    return float(np.median(values)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def pairwise_disagreement(metrics, rules, horizon_index):
    endpoint = np.asarray([metrics["endpoint"][rule][horizon_index] for rule in rules])
    mismatch = np.asarray([metrics["mismatch"][rule][horizon_index] for rule in rules])
    flips = sum(
        1 for left in range(len(rules)) for right in range(left + 1, len(rules))
        if ((endpoint[left] < endpoint[right])
            != (mismatch[left] < mismatch[right])))
    return flips / (len(rules) * (len(rules) - 1) / 2)


def evaluate(results_path):
    with open(results_path, "rb") as handle:
        blob = pickle.load(handle)
    cfg, records = blob["cfg"], blob["records"]
    horizons = cfg["horizons"]
    rules = (["refit", "replay"]
             + [f"proximal-{mu:g}" for mu in cfg["update_rules"]["proximal"]["mus"]])
    hidx = np.asarray(horizons)
    P0E = env_panel_predictions(cfg, "v0")[hidx].reshape(len(horizons), -1)
    P1E = env_panel_predictions(cfg, "v1")[hidx].reshape(len(horizons), -1)

    retained, accounting = apply_exclusions(cfg, records)
    per_n0 = {}
    for record in retained:
        metrics = compute_seed_metrics(record, P0E, P1E, rules, horizons)
        per_n0.setdefault(record["n0"], []).append((record["seed"], metrics))
    for values in per_n0.values():
        values.sort(key=lambda pair: pair[0])

    output_dir = os.path.dirname(results_path)
    repetitions = cfg["corrected_design"]["bootstrap_repetitions"]
    rng = np.random.default_rng(cfg["master_seed"] + 17)
    tolerances = cfg["corrected_design"]["tie_relative_tolerances"]

    metric_rows = []
    cosine_rows = []
    directional_rows = []
    summary = ["# Corrected paired neural experiment", "", "## Accounting", "",
               "| N0 | attempted | retained | MSE exclusions | divergence exclusions |",
               "|---:|---:|---:|---:|---:|"]
    for n0 in sorted(per_n0):
        account = accounting[n0]
        summary.append(f"| {n0} | {account['attempted']} | {account['retained']} | "
                       f"{account['excluded_mse']} | {account['excluded_diverged']} |")

    summary += ["", "## Primary directional result", "",
                "Expected direction means refit is better by endpoint and proximal-mu=1 is better by mismatch.", "",
                "| N0 | inherited h=1 | expected / n | opposite / n | pairwise flips | exact best disagreement | 5% tie-aware no-overlap |",
                "|---:|---:|---:|---:|---:|---:|---:|"]

    for n0 in sorted(per_n0):
        pairs = per_n0[n0]
        metrics_list = [metrics for _, metrics in pairs]
        inherited = np.median(np.stack([m["inherited"] for m in metrics_list]), axis=0)

        for rule in rules:
            endpoint = np.median(np.stack([m["endpoint"][rule] for m in metrics_list]), axis=0)
            mismatch = np.median(np.stack([m["mismatch"][rule] for m in metrics_list]), axis=0)
            cosines = np.asarray([m["cos_h1"][rule] for m in metrics_list])
            cosine_rows.append([n0, rule, float(np.median(cosines)),
                                float(np.quantile(cosines, 0.25)),
                                float(np.quantile(cosines, 0.75))])
            for index, horizon in enumerate(horizons):
                metric_rows.append([n0, rule, horizon, endpoint[index],
                                    inherited[index], mismatch[index], len(metrics_list)])

        for index, horizon in enumerate(horizons):
            expected = opposite = binary = 0
            pairwise = []
            no_overlap = {tol: 0 for tol in tolerances}
            endpoint_differences = []
            mismatch_differences = []
            for metrics in metrics_list:
                endpoint = np.asarray([metrics["endpoint"][rule][index] for rule in rules])
                mismatch = np.asarray([metrics["mismatch"][rule][index] for rule in rules])
                refit_index = rules.index("refit")
                prox_index = rules.index("proximal-1")
                e_refit, e_prox = endpoint[refit_index], endpoint[prox_index]
                m_refit, m_prox = mismatch[refit_index], mismatch[prox_index]
                expected += int(e_refit < e_prox and m_refit > m_prox)
                opposite += int(e_refit > e_prox and m_refit < m_prox)
                endpoint_differences.append(e_prox - e_refit)
                mismatch_differences.append(m_refit - m_prox)
                binary += int(np.argmin(endpoint) != np.argmin(mismatch))
                pairwise.append(pairwise_disagreement(metrics, rules, index))
                for tolerance in tolerances:
                    endpoint_set = set(np.flatnonzero(
                        endpoint <= endpoint.min() * (1.0 + tolerance)))
                    mismatch_set = set(np.flatnonzero(
                        mismatch <= mismatch.min() * (1.0 + tolerance)))
                    no_overlap[tolerance] += int(not endpoint_set.intersection(mismatch_set))

            n = len(metrics_list)
            expected_lo, expected_hi = wilson(expected, n)
            opposite_lo, opposite_hi = wilson(opposite, n)
            endpoint_med, endpoint_lo, endpoint_hi = bootstrap_median_ci(
                endpoint_differences, repetitions, rng)
            mismatch_med, mismatch_lo, mismatch_hi = bootstrap_median_ci(
                mismatch_differences, repetitions, rng)
            row = [n0, horizon, n, inherited[index], expected, expected / n,
                   expected_lo, expected_hi, opposite, opposite / n,
                   opposite_lo, opposite_hi, (expected + opposite) / n,
                   float(np.mean(pairwise)), binary / n,
                   endpoint_med, endpoint_lo, endpoint_hi,
                   mismatch_med, mismatch_lo, mismatch_hi]
            row.extend(no_overlap[tol] / n for tol in tolerances)
            directional_rows.append(row)

            if horizon == 1:
                summary.append(
                    f"| {n0} | {inherited[index]:.4f} | {expected}/{n} | "
                    f"{opposite}/{n} | {100*np.mean(pairwise):.1f}% | "
                    f"{100*binary/n:.1f}% | {100*no_overlap[0.05]/n:.1f}% |")

    # Paired N0 audit: all conditions for a seed share D1 and use nested D0.
    n0_grid = sorted(per_n0)
    by_seed = {
        seed: {n0: metrics for n0 in n0_grid
               for candidate_seed, metrics in per_n0[n0]
               if candidate_seed == seed}
        for seed in sorted({seed for values in per_n0.values() for seed, _ in values})
    }
    paired_drops = []
    within_correlations = []
    strictly_decreasing_inherited = 0
    high_exceeds_low = 0
    for seed, condition_metrics in by_seed.items():
        if len(condition_metrics) != len(n0_grid):
            continue
        inherited = np.asarray([condition_metrics[n0]["inherited"][0] for n0 in n0_grid])
        pairwise = np.asarray([
            pairwise_disagreement(condition_metrics[n0], rules, 0) for n0 in n0_grid])
        strictly_decreasing_inherited += int(np.all(np.diff(inherited) < 0))
        paired_drops.append(pairwise[0] - pairwise[-1])
        high_exceeds_low += int(pairwise[0] > pairwise[-1])
        correlation = float(np.corrcoef(inherited, pairwise)[0, 1])
        if np.isfinite(correlation):
            within_correlations.append(correlation)

    paired_drops = np.asarray(paired_drops)
    bootstrap_ids = rng.integers(
        0, len(paired_drops), size=(repetitions, len(paired_drops)))
    bootstrap_means = np.mean(paired_drops[bootstrap_ids], axis=1)
    paired_mean = float(np.mean(paired_drops))
    paired_lo, paired_hi = np.quantile(bootstrap_means, [0.025, 0.975])
    median_correlation = float(np.median(within_correlations))
    positive_correlations = int(np.sum(np.asarray(within_correlations) > 0))
    n_paired = len(paired_drops)
    summary += [
        "", "## Within-seed paired audit", "",
        f"- Inherited error is strictly decreasing across all five N0 prefixes "
        f"for {strictly_decreasing_inherited}/{n_paired} seeds.",
        f"- Pairwise disagreement is higher at N0={n0_grid[0]} than at "
        f"N0={n0_grid[-1]} for {high_exceeds_low}/{n_paired} seeds.",
        f"- The mean paired drop is {100*paired_mean:.1f} percentage points "
        f"(bootstrap 95% interval [{100*paired_lo:.1f}, {100*paired_hi:.1f}]).",
        f"- Median within-seed correlation between inherited error and pairwise "
        f"disagreement is {median_correlation:.3f}; "
        f"{positive_correlations}/{len(within_correlations)} correlations are positive.",
    ]

    write_csv(
        os.path.join(output_dir, "metrics.csv"),
        ["n0", "rule", "horizon", "median_endpoint", "median_inherited",
         "median_mismatch", "n_retained"], metric_rows)
    write_csv(
        os.path.join(output_dir, "cosines.csv"),
        ["n0", "rule", "median_cos_h1", "q25", "q75"], cosine_rows)
    directional_header = [
        "n0", "horizon", "n_retained", "median_inherited", "expected_count",
        "expected_rate", "expected_wilson_lo", "expected_wilson_hi",
        "opposite_count", "opposite_rate", "opposite_wilson_lo",
        "opposite_wilson_hi", "total_flip_rate", "mean_pairwise_disagreement",
        "exact_best_disagreement", "median_endpoint_diff_prox_minus_refit",
        "endpoint_diff_boot_lo", "endpoint_diff_boot_hi",
        "median_mismatch_diff_refit_minus_prox", "mismatch_diff_boot_lo",
        "mismatch_diff_boot_hi"]
    directional_header.extend(f"no_overlap_rel_tol_{tol:g}" for tol in tolerances)
    write_csv(os.path.join(output_dir, "directional.csv"),
              directional_header, directional_rows)
    write_csv(
        os.path.join(output_dir, "paired_audit.csv"),
        ["n_paired", "strictly_decreasing_inherited", "high_n0min_pairwise_gt_n0max",
         "mean_pairwise_drop", "bootstrap_lo", "bootstrap_hi",
         "median_within_seed_correlation", "positive_correlations"],
        [[n_paired, strictly_decreasing_inherited, high_exceeds_low,
          paired_mean, paired_lo, paired_hi, median_correlation,
          positive_correlations]])

    with open(os.path.join(output_dir, "SUMMARY.md"), "w") as handle:
        handle.write("\n".join(summary) + "\n")

    make_figure(output_dir, directional_rows, horizons)
    print("\n".join(summary))
    print(f"\nWrote corrected evaluation to {output_dir}")


def make_figure(output_dir, rows, horizons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h1 = [row for row in rows if row[1] == 1]
    h1.sort(key=lambda row: row[3])
    x = np.asarray([row[3] for row in h1])
    expected = np.asarray([100 * row[5] for row in h1])
    lower = np.asarray([100 * row[6] for row in h1])
    upper = np.asarray([100 * row[7] for row in h1])
    pairwise = np.asarray([100 * row[13] for row in h1])

    fig, axis = plt.subplots(figsize=(4.2, 3.0))
    axis.plot(x, expected, marker="o", color="#0072B2",
              label="directional refit--proximal reversal")
    axis.fill_between(x, lower, upper, color="#0072B2", alpha=0.15, linewidth=0)
    axis.plot(x, pairwise, marker="s", color="#D55E00",
              label="mean pairwise rank flips")
    axis.set_xlabel("measured inherited error at $h=1$")
    axis.set_ylabel("disagreement (%)")
    axis.set_ylim(-2, 102)
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "corrected_dose_response.pdf"))
    fig.savefig(os.path.join(output_dir, "corrected_dose_response.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    default = os.path.join(BASE, "results_corrected", "results_v1.pkl")
    evaluate(sys.argv[1] if len(sys.argv) > 1 else default)
