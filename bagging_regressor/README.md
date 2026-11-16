# Bagging regressor, from scratch

A bagging (bootstrap-aggregating) regressor built from decision stumps —
no scikit-learn, just numpy. Each stump is a single-feature, single-threshold
split that predicts the mean of its leaf. Fifty stumps are each trained on a
bootstrap resample of the data (sampling with replacement), and the ensemble
predicts the mean of the stumps' predictions.

## Why

A single decision stump has high variance: resample the data slightly and
its split — and therefore its predictions — can jump around. Averaging many
stumps trained on different resamples cancels out that noise, so the
ensemble's test error is lower than a single stump's. This is the core idea
behind bagging, and behind random forests (which add a random feature subset
per tree — supported here with `max_features`).

## Usage

```python
from bagging_regressor.model import BaggingRegressor, DecisionStumpRegressor

model = BaggingRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

`BaggingRegressor(n_estimators=50, random_state=None, max_features=None)`:
`n_estimators` is how many stumps to average; `random_state` seeds the
bootstrap sampling; `max_features` limits each stump to a random subset of
that many features. After fitting, `model.estimators_` holds the fitted
stumps.

## Demo

```bash
python bagging_regressor/demo.py
```

Generates 200 noisy sine points (`y = sin(x) + noise`, sigma 0.3, seed 42),
fits a single stump and a 50-stump bagged ensemble on an 80/20 split, and
prints both test MSEs plus the bagged model's R². Bagging should clearly
beat the single stump.

## Run tests

```bash
python -m pytest tests/test_bagging_regressor.py -v
```
