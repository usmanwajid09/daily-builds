# Huber Regression

Robust linear regression with the **Huber loss**, fitted by iteratively
reweighted least squares (IRLS). Unlike OLS, which squares every residual
and lets a few gross outliers drag the whole fit, the Huber loss is
quadratic near zero and linear far away — outliers are down-weighted
instead of dominating.

## Why it's useful

- Data with measurement errors, sensor glitches, or mislabeled targets.
- Any regression where a handful of bad points shouldn't wreck the line.
- A drop-in replacement for OLS when you suspect contamination.

## How it works

Start from the OLS solution, then repeat:

1. Compute residuals `r = y - Xθ`.
2. Estimate a robust scale `σ = 1.4826 · MAD(r)` (floored at `1e-8`), so
   `delta` is in units of robust sigma and behaves sensibly at any noise scale.
3. Weights `w = 1` if `|r|/σ ≤ δ`, else `w = δ·σ/|r|`.
4. Re-solve weighted least squares on the `√w`-scaled design (with intercept).
5. Stop when the coefficients stop changing (`tol`) or `n_iter` is hit.

Default `delta=1.35` gives ~95% efficiency at the Gaussian.

## Usage

```python
from huber_regression import HuberRegression, ols_fit

model = HuberRegression(delta=1.35).fit(X, y)
y_pred = model.predict(X_new)

print(model.coef_, model.intercept_)   # fitted parameters
print(model.weights_)                  # final IRLS weights (< 1 = down-weighted)
print(model.n_iter_run_)               # iterations actually run

coef, intercept = ols_fit(X, y)        # plain OLS, for comparison
```

## Run the demo

```bash
python huber_regression/demo.py
```

Generates `y = 2 + 3x + noise`, corrupts 10% of targets with ±15 outliers,
and shows Huber beating OLS on clean test data.

## Run the tests

```bash
python -m pytest tests/test_huber_regression.py -v
```
