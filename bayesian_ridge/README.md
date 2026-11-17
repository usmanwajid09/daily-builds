# Bayesian Ridge Regression (from scratch)

Bayesian linear regression with a Gaussian prior and a closed-form
posterior — numpy only, no new dependencies.

## What it does

Ordinary least squares gives you one best-fit line and no sense of how
sure it is. This regressor instead treats the weights as random variables:

- **Prior:** `w ~ N(0, alpha^-1 I)` — weights start near zero; `alpha` (the
  prior precision) controls how strongly they are pulled there.
- **Likelihood:** `y = X @ w + eps`, `eps ~ N(0, beta^-1)` — `beta` (the
  noise precision) says how much to trust the data.
- **Posterior** (Gaussian, closed form — no sampling, no iteration):
  `S = inv(alpha * I + beta * X^T X)` and `m = beta * S @ X^T y`.

The intercept is handled by centering: `X` and `y` are demeaned before
fitting and the intercept is recovered as `y_mean - X_mean @ coef_`, so
the prior regularises the slopes only, not the offset.

## Why predictive uncertainty is useful

`predict(X)` returns the posterior predictive mean; `predict_std(X)`
returns its standard deviation, `sqrt(1/beta + x^T S x)`. The first term
is irreducible observation noise; the second is parameter uncertainty,
which grows as a point moves away from the training data. That means the
model can say "I don't know" — flagging out-of-distribution inputs,
driving active learning (query the most uncertain points), or feeding
downstream decisions a confidence interval instead of a bare number.

## How to run

```bash
# from the repo root
python bayesian_ridge/demo.py
python -m pytest tests/test_bayesian_ridge.py -q
```

## Demo output

Fitting `y = 2x - 1 + noise` (seed 42, n=100, 80/20 split):

```
Learned coefficient: 1.9838 (true: 2.0)
Learned intercept:   -1.0360 (true: -1.0)
Test R^2:            0.9769
Test MSE:            0.2622
Mean predictive std: 1.0068
All checks passed.
```

## Tests

11 pytest tests in `tests/test_bayesian_ridge.py`: coefficient recovery on
noiseless data, R² > 0.9 on noisy data, predictive std positive and growing
with distance from the training data, posterior covariance symmetric
positive-definite, predict shape, and `ValueError`s for 1D X, mismatched
lengths, empty input, NaN input, and `alpha`/`beta <= 0`.
