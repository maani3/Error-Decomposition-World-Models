# Decomposing Error in Continual World Models — paper, code, and data

Camera-ready paper accepted to the NeurIPS 2026 Workshop on Continual World Models
(Idea Track): [read the PDF](paper.pdf).

Everything reported in the paper is produced by the scripts here; no result is
transcribed by hand.

The paper decomposes post-update ("endpoint") prediction error into inherited
error and update mismatch through the pointwise identity

```
eps^1  =  eps^0  +  (delta_M - delta_E)
```

where `delta_M` and `delta_E` are the model's and the environment's version
deltas on the same version-comparable query. Two settings instantiate it: a
linear structural dynamical system where `delta_E` is available in closed form,
and a learned MLP world model on a versioned nonlinear simulator.

## Layout

| Path | What it is |
|---|---|
| `delta_contract_experiment.py` | Linear system: identity check, the two base-quality regimes, the inherited-error sweep, the spectral sweep. Writes `results.{md,json}` and figures 1–3. |
| `followup_experiments.py` | E5–E10: K-round accumulation, nonlinear rollout, aggregation conventions, Wilson intervals, the non-oracle proximal ablation, cancellation geometry. Writes `results_followup.{md,json}` and figures 4–5. |
| `make_combined_paper_figure.py` | Builds the paper's three-panel main figure from the linear run plus `neural_extension/results_corrected/directional.csv`. |
| `neural_extension/` | Learned neural world model on a versioned damped pendulum. |
| `results*.json`, `results*.md` | Committed outputs of the two linear scripts, so every number in the paper can be checked without re-running. `results.json` is also a required **input** to `followup_experiments.py` (E8 reads the headline counts from it). |

Inside `neural_extension/`:

| Path | What it is |
|---|---|
| `config_corrected.yaml` | Every predeclared choice — panel, horizons, thresholds, training rule, update rules, exclusions — frozen before the run. |
| `simulator.py` | Versioned pendulum (RK4, `dt=0.05`; version change `g: 9.8 -> 11.5`), exact environment rollouts, data generation. |
| `model.py`, `train.py` | 2–64–64–2 tanh delta-MLP; training with early stopping, replay mixture, L2-SP proximal fine-tuning. |
| `run_corrected.py` | The paired sweep: one maximal `D0` per seed with nested `N0` prefixes, `D1` and the refit model shared across conditions, trajectory-level validation. |
| `evaluate.py` | Shared metric library (signed-vector metrics, predeclared exclusions, Wilson intervals). Imported by `evaluate_corrected.py`. |
| `evaluate_corrected.py` | Entry point for evaluation: writes the CSVs, `SUMMARY.md`, and the dose–response figure. |
| `results_corrected/` | Committed evaluation outputs and the run's environment record. |
| `tests/` | Unit tests: the identity holds pointwise for real trained networks; simulator energy is non-increasing under damping; `delta_E` is nonzero on every panel query; the nested-prefix/validation split is structurally correct. |

## Reproducing

```bash
pip install -r requirements.txt

# Linear results (numpy only, runs in seconds)
python delta_contract_experiment.py           # add --nofig to skip figures
python followup_experiments.py                # reads results.json

# Neural results (CPU torch, ~12 min on 10 single-threaded workers)
cd neural_extension
python tests/test_units.py
python tests/test_corrected_design.py
python run_corrected.py full                  # writes results_corrected/results_v1.pkl
python evaluate_corrected.py                  # regenerates the committed CSVs

# The paper's main figure
cd .. && python make_combined_paper_figure.py
```

The raw neural predictions (`results_corrected/results_v1.pkl`, ~5 MB) are not
distributed; `run_corrected.py full` regenerates them deterministically. The
committed CSVs in `results_corrected/` are sufficient to check every neural
number in the paper and to build the main figure without running torch.

## Determinism

All randomness derives from `master_seed` in the config via
`np.random.SeedSequence`; each training run gets its own torch seed controlling
initialization, split, and minibatch order. Workers are single-threaded
(`torch.set_num_threads(1)`), so runs are bit-reproducible. A seed re-run from
data generation through every update rule reproduced the stored record exactly,
including epoch counts. Environment of the reported run is in
`neural_extension/results_corrected/ENVIRONMENT.md`.

## Reading the results

Seed accounting is reported rather than assumed. The linear experiments attempt
600 seeded systems and retain 544 after predeclared stability and propagation
checks; the neural sweep retained all 500 condition records. The identity is
asserted pointwise at runtime for every retained (seed, rule, horizon) in
`evaluate.py`, in addition to the unit test.

Two conventions matter when reading the tables. `adapter` in the linear
experiments is an **oracle diagnostic**, not a proposed algorithm: it is told
which transition entry changed, and exists only to instantiate a faithful
revision of a biased model. It must not be read as a baseline that wins. And
all three reported quantities use separate relative normalizations, so their
numerical values are not directly comparable across columns.
