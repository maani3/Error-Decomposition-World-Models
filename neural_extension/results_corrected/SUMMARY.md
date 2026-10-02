# Corrected paired neural experiment

## Accounting

| N0 | attempted | retained | MSE exclusions | divergence exclusions |
|---:|---:|---:|---:|---:|
| 5 | 100 | 100 | 0 | 0 |
| 15 | 100 | 100 | 0 | 0 |
| 40 | 100 | 100 | 0 | 0 |
| 150 | 100 | 100 | 0 | 0 |
| 500 | 100 | 100 | 0 | 0 |

## Primary directional result

Expected direction means refit is better by endpoint and proximal-mu=1 is better by mismatch.

| N0 | inherited h=1 | expected / n | opposite / n | pairwise flips | exact best disagreement | 5% tie-aware no-overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.0380 | 95/100 | 0/100 | 83.2% | 99.0% | 99.0% |
| 15 | 0.0142 | 64/100 | 1/100 | 61.2% | 96.0% | 90.0% |
| 40 | 0.0092 | 27/100 | 1/100 | 31.5% | 74.0% | 55.0% |
| 150 | 0.0054 | 12/100 | 3/100 | 11.5% | 50.0% | 27.0% |
| 500 | 0.0041 | 6/100 | 1/100 | 7.6% | 31.0% | 20.0% |

## Within-seed paired audit

- Inherited error is strictly decreasing across all five N0 prefixes for 70/100 seeds.
- Pairwise disagreement is higher at N0=5 than at N0=500 for 99/100 seeds.
- The mean paired drop is 75.6 percentage points (bootstrap 95% interval [72.4, 78.4]).
- Median within-seed correlation between inherited error and pairwise disagreement is 0.910; 100/100 correlations are positive.
