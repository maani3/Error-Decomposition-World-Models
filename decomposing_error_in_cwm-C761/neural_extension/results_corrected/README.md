# Corrected paired neural extension

This directory contains the neural experiment reported in the paper.  It
supersedes an earlier independent-condition run, whose comparability problems
are listed below; only the corrected paired design is distributed here.

## What was corrected

For each of 100 paired seeds, the experiment generates one 500-trajectory D0
dataset and uses literal nested prefixes at N0 = 5, 15, 40, 150, and 500.  All
five conditions share the same 40-trajectory D1 dataset.  Validation holds out
whole trajectories (indices 0, 10, 20, ...) instead of correlated individual
transitions.  Base-network initialization is shared across N0, the refit model
is trained once per seed, and proximal variants share validation data and
minibatch randomness.  These choices remove the main comparability and leakage
concerns in the original run.

The simulator, neural architecture, observation noise, query panel, horizons,
and update rules are otherwise unchanged.  The complete specification is in
`../config_corrected.yaml`; implementation is in `../run_corrected.py` and
`../evaluate_corrected.py`.

## Primary h=1 results

Expected direction means refit is better by endpoint while proximal-mu=1 is
better by mismatch.

| N0 | inherited error | expected | opposite | mean pairwise flips | 5%-tie-aware best-set disagreement |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.0380 | 95/100 | 0/100 | 83.2% | 99.0% |
| 15 | 0.0142 | 64/100 | 1/100 | 61.2% | 90.0% |
| 40 | 0.0092 | 27/100 | 1/100 | 31.5% | 55.0% |
| 150 | 0.0054 | 12/100 | 3/100 | 11.5% | 27.0% |
| 500 | 0.0041 | 6/100 | 1/100 | 7.6% | 20.0% |

At N0=5, refit has median endpoint/mismatch 0.010/1.064, compared with
0.035/0.557 for proximal-mu=1.  The expected reversal occurs on 95/100 seeds
(Wilson 95% interval [88.8, 97.8]) and on 81/100 seeds at h=20.  The paired
median endpoint gap (proximal minus refit) is 0.0257 with bootstrap interval
[0.0198, 0.0298]; the mismatch gap (refit minus proximal) is 0.5456 with
interval [0.4697, 0.6319].

## Paired dose-response audit

- Pairwise disagreement is higher at N0=5 than N0=500 on 99/100 seeds.
- The mean paired drop is 75.6 percentage points, with bootstrap 95% interval
  [72.4, 78.4].
- The median within-seed correlation between inherited error and pairwise
  disagreement is 0.910; all 100 correlations are positive.
- Median inherited error is monotone across N0.  Individual inherited-error
  sequences are strictly monotone on 70/100 seeds, reflecting residual
  optimization and observation noise.
- All 500 attempted condition records were retained; none diverged or failed
  the declared base-MSE exclusion.

These results support a strong association between inherited error and ranking
disagreement under a paired manipulation.  They do not establish inherited
error as the only cause, because increasing N0 changes the base training set in
order to change inherited error.

## Verification and files

- `results_v1.pkl`: raw signed predictions and embedded configuration.
- `directional.csv`: directional, opposite, pairwise, tie-aware, Wilson, and
  paired-bootstrap statistics at every horizon.
- `metrics.csv`: median endpoint, inherited, and mismatch metrics.
- `cosines.csv`: cancellation geometry.
- `paired_audit.csv`: within-seed dose-response audit.
- `corrected_dose_response.{pdf,png}`: paper-ready h=1 figure.
- `SUMMARY.md`: generated concise tables.
- `ENVIRONMENT.md`: exact software and execution record.

The structural tests and original simulator/identity tests pass.  Seed 0 was
rerun from data generation through all five conditions and every update rule;
it was bit-identical to the stored result, including epoch counts and predictions.
