# Theil-Sen Regressor

Robust linear regression using the **Theil-Sen estimator**: the slope is the
median of all pairwise slopes `(y_j - y_i) / (x_j - x_i)`, and the intercept
is the median of `y - slope * x`. Because it uses medians instead of means,
a minority of gross outliers barely moves the fit (breakdown point ~29%),
unlike ordinary least squares.

## How it works

- **Single feature:** exact median of all pairwise slopes.
- **Multivariate (d > 1):** per-feature Theil-Sen slopes, refined by robust
  backfitting (each feature re-fit on its partial residual, iterated to
  convergence), then intercept = median residual. **Limitation:** this is
  an approximation of a true multivariate Theil-Sen fit, not the exact
  spatial-median estimator.
- **Large n:** set `max_subpopulation` to subsample pairs deterministically
  (seeded by `random_state`) instead of computing all O(n^2) slopes.

## Usage

```python
from theil_sen_regressor import TheilSenRegressor

model = TheilSenRegressor().fit(X, y)
print(model.coef_, model.intercept_)
print(model.predict(X_new))
print(model.score(X, y))  # R^2
```

## Run

```bash
python theil_sen_regressor/demo.py
pytest tests/test_theil_sen_regressor.py -v
```

Numpy only — no sklearn/scipy.
