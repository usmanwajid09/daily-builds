# Voting Regressor

An ensemble regressor that combines three small, from-scratch (numpy-only,
no sklearn) base learners — **Ridge regression**, **k-nearest neighbours**
and a **CART decision tree** — and merges their predictions with a voting
rule.

## What it does

`VotingRegressor` fits all three base learners on the full training data,
then combines their per-sample predictions with one of three rules:

| Voting | How predictions are combined |
|---|---|
| `"mean"` | Plain average of the three base predictions |
| `"median"` | Per-sample median of the three base predictions |
| `"weighted"` | Weighted average using `weights` (normalised to sum to 1) |

`weights` is a list/array of 3 non-negative numbers (one per base learner)
or `None` for equal weights. `"mean"` and `"median"` ignore `weights`.
`random_state` (default 42) is used only for reproducibility of any
tie-breaking.

## How it works

- **RidgeRegressor** — closed-form ridge regression with an L2 penalty
  (`alpha`) and an optional intercept. The intercept is not penalised.
- **KNNRegressor** — k-nearest-neighbours regression (`n_neighbors=5`)
  with Euclidean distance and uniform weights.
- **DecisionTreeRegressor** — a small CART tree that picks the split
  (feature, threshold) minimising total within-node sum of squared errors,
  controlled by `max_depth` and `min_samples_split`. Split ties are broken
  deterministically, so results are reproducible.

### Why median voting is robust

With three base learners, the per-sample median discards the highest and
lowest prediction and keeps the middle one. If one learner goes badly
wrong on a sample — say a tree that overfits a noisy region — the median
ignores it entirely, while the mean is dragged a third of the way toward
the bad prediction. The test suite includes a check for exactly this:
corrupt one base learner to always predict 1000 and median voting stays
close to the truth while mean voting is pulled far away.

## How to run

From the repo root:

```bash
python voting_regressor/demo.py
```

Run the tests:

```bash
python -m pytest tests/test_voting_regressor.py -q
```

## Example output

```
Train size: 160, test size: 40
Single Ridge baseline test RMSE: 1.4599
VotingRegressor (mean) test RMSE: 0.6924
VotingRegressor (median) test RMSE: 0.6119
VotingRegressor (weighted [0.5, 0.3, 0.2]) test RMSE: 0.8580
First 5 test samples — base predictions [Ridge, k-NN, Tree] vs mean vote:
  true=-0.934  base=[ 0.833 -0.779 -0.045]  mean=+0.003
  true=-0.964  base=[ 1.491 -0.731 -0.522]  mean=+0.080
  true=-0.420  base=[ 0.916 -0.162 -0.045]  mean=+0.236
  true=-0.406  base=[ 0.96  -0.179 -0.045]  mean=+0.245
  true=+1.756  base=[1.141 1.219 0.892]  mean=+1.084
```

All three voting rules beat the single Ridge baseline (RMSE 1.46) on this
synthetic `y = sin(3·x0) + x1² + noise` task; median voting is best here
because the linear Ridge learner is the weak link on this non-linear
target, and the median limits how much it can hurt.
```

## Files

- `model.py` — `RidgeRegressor`, `KNNRegressor`, `DecisionTreeRegressor`,
  `VotingRegressor` (numpy only).
- `__init__.py` — exposes `VotingRegressor`.
- `demo.py` — synthetic-data demo comparing the three voting rules
  against a single Ridge baseline.
