# Stacking Regressor

Stacked regression built from scratch with numpy only — no sklearn.

## What it does

Three base regressors — ridge regression, k-nearest neighbours and a small
CART decision tree — each predict the target, and a ridge meta-learner
combines their predictions into the final output. On a non-linear target
such as `y = sin(3·x0) + x1² + noise`, the stacker clearly beats a single
linear ridge model, because the tree and k-NN base learners can capture
curves the linear model cannot.

## How it works

1. **Out-of-fold meta-features.** The training data is split into `n_folds`
   shuffled folds. Each base learner is trained on K−1 folds and predicts
   the held-out fold, so every training sample gets a prediction from a
   model that never saw it. The three predictions per sample form the
   meta-features, a matrix of shape `(n_samples, 3)`.
2. **Ridge meta-learner.** A ridge regression (with intercept) is fitted on
   the meta-features. Its weights — `[intercept, w_ridge, w_knn, w_tree]` —
   say how much to trust each base learner.
3. **Refit on full data.** Each base learner is refit on the full training
   set, and at predict time their predictions are fed through the
   meta-learner.

**Why out-of-fold avoids leakage:** if the meta-learner were trained on
in-sample predictions, a flexible base learner (the tree) would look almost
perfect on data it already memorised and would grab nearly all the weight,
even though its out-of-sample predictions are far worse. Out-of-fold
predictions are honest estimates of each learner's real skill, so the
meta-learner weights reflect genuine performance.

## Files

- `model.py` — `RidgeRegressor`, `KNNRegressor`, `DecisionTreeRegressor`
  (all written in-file) and `StackingRegressor` (`n_folds=5`, `alpha=1.0`,
  `random_state=42`).
- `demo.py` — synthetic non-linear demo vs a ridge baseline.
- `__init__.py` — exposes `StackingRegressor`.

## How to run

From the repo root:

```bash
python stacking_regressor/demo.py
```

Run the tests:

```bash
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_stacking_regressor.py -q
```

## Example output

```
Stacking regressor demo (y = sin(3*x0) + x1^2 + noise)
  train: 160 samples, test: 40 samples
  StackingRegressor RMSE: 0.3154
  RidgeRegressor  RMSE:   1.4599
  Meta-learner weights [intercept, ridge, knn, tree]: [0.067, -0.1556, 0.9472, 0.1742]
  Stacker beats ridge by 1.1444 RMSE (78.4% lower)
```

Note the meta-learner puts most weight on k-NN here (0.95) — on this smooth
low-dimensional target it is the strongest base learner — while the ridge
weight goes slightly negative, correcting the linear model's bias.
