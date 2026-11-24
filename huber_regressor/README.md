# Huber Regressor

Robust linear regression with the Huber loss, fitted by iteratively
reweighted least squares (IRLS). NumPy + standard library only.

## The model

Ordinary least squares minimises the sum of squared residuals, so one
gross outlier can drag the fitted line far off. The Huber loss is
quadratic for small residuals and linear for large ones, capping the
influence of any single point:

```
L(r) = 0.5 * r^2                 if |r| <= c
L(r) = c * |r| - 0.5 * c^2       if |r| >  c
```

where `c = epsilon * scale` and `scale` is a robust estimate of the
residual spread (`1.4826 * MAD`, the median absolute deviation, which
matches the standard deviation for Gaussian noise). The default
`epsilon = 1.35` gives 95% efficiency at the Gaussian model while still
down-weighting outliers.

Fitting uses IRLS, starting from the OLS fit:

1. residuals `r = y - y_pred`
2. `scale = 1.4826 * median(|r - median(r)|)` (falls back to the mean
   absolute deviation, then to 1.0, if the MAD is 0, e.g. a perfect fit)
3. weights `w = 1` if `|r| <= epsilon * scale`, else
   `w = epsilon * scale / |r|`
4. solve the weighted least-squares problem with those weights

Repeat until the largest parameter change is below `tol` or `max_iter`
is reached. Each weighted step is solved with `numpy.linalg.lstsq` on
the sqrt-weight-scaled design matrix.

The intercept is fitted by augmenting the design matrix with a column
of ones when `fit_intercept=True`; with `fit_intercept=False` the line
goes through the origin and `intercept_` is 0.0.

## Usage

```python
from huber_regressor.model import HuberRegressor

model = HuberRegressor(epsilon=1.35, max_iter=100, tol=1e-6, fit_intercept=True)
model.fit(X_train, y_train)

print(model.coef_)       # slope(s)
print(model.intercept_)  # intercept
print(model.n_iter_)     # IRLS iterations used
print(model.predict(X_test))
print(model.score(X_test, y_test))  # R^2
```

`fit` raises `ValueError` for non-2D `X`, mismatched `X`/`y` lengths,
empty input, or non-finite values; the constructor raises `ValueError`
for `epsilon <= 0`, `max_iter < 1`, or `tol <= 0`. `predict` before
`fit` raises `ValueError`.

## Demo

`demo.py` generates `y = 2x - 1 + noise` (seed 42), corrupts ~10% of
the training points with gross outliers (+8 to +12), and fits both
`HuberRegressor` and plain OLS (`numpy.linalg.lstsq`) for comparison on
a clean test set:

```
True slope / intercept:     2.0000 / -1.0000
Huber slope / intercept:    2.0179 / -0.9191  (IRLS iterations: 9)
OLS slope / intercept:      1.9315 / -0.1201
Huber test MSE:             0.2565
OLS test MSE:               0.9546
Huber slope error:          0.0179
OLS slope error:            0.0685
OK: Huber beats OLS on outlier-contaminated data.
```

Run it with:

```bash
~/workspace/db-venv/bin/python ~/workspace/daily-builds/huber_regressor/demo.py
```

## Tests

16 pytest tests in `tests/test_huber_regressor.py` cover: coefficient
recovery on clean noiseless data (single- and multi-feature), R² > 0.9
on noisy data, beating OLS on outlier-contaminated data, predict shape,
score in (-inf, 1], `fit_intercept=False`, `ValueError`s for 1D `X`,
mismatched lengths, empty input, NaN, `epsilon <= 0`, predict before
fit, determinism, fitted attributes, and agreement with OLS when the
data is clean.

```bash
cd ~/workspace/daily-builds
~/workspace/db-venv/bin/python -m pytest tests/test_huber_regressor.py -q
# 16 passed
```
