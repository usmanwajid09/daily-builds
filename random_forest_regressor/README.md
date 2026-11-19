# Random Forest Regressor

Random-forest regression from scratch (numpy only). Each tree is a
CART regression tree (variance reduction, greedy splits) grown on a
bootstrap resample, with a random subset of features at each split.
Predictions are the mean of the trees' predictions, and samples left
out of each bootstrap sample give an out-of-bag (OOB) R² estimate
without a held-out test set.

## Run

```bash
python random_forest_regressor/demo.py
python -m pytest tests/test_random_forest_regressor.py -q
```

## Example

Demo on y = 10·sin(x0) + 5·x1² + noise (x2 is pure noise), 30 trees:

```
Samples: 200, features: 3, trees: 30
Train R^2: 0.9901
OOB R^2:   0.9494
Feature importances:
  x0: 0.203
  x1: 0.787
  x2: 0.010
```

The noise feature x2 gets almost no splits, as it should.

## API

- `RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, max_features="sqrt", random_state=None)`
- `fit(x, y)` → self
- `predict(x)` → mean prediction per row
- `score(x, y)` → R²
- `.oob_score_` → R² of the out-of-bag predictions
- `.feature_importances_` → fraction of splits per feature
