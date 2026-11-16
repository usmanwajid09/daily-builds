# Polynomial Regression

Single-feature polynomial regression with ridge regularisation, fitted by
the closed-form solution `(X^T X + alpha * I)^-1 X^T y` (intercept never
penalised). The input is standardised inside `fit` (train mean/std) so
higher degrees stay numerically stable, and the fitted coefficients are
converted back to the original `x` basis — `coef_[0]` is the intercept and
`coef_[k]` multiplies `x**k`.

## Why it's useful

Real relationships are rarely straight lines: a quadratic or cubic often
captures the curve. Plain least squares with high-degree features is
unstable and overfits, so ridge shrinkage plus a validation-based degree
choice (`select_degree`) keeps the model honest.

## Usage

```python
import numpy as np
from polynomial_regression.model import PolynomialRegression, select_degree

rng = np.random.default_rng(42)
x = rng.uniform(-3, 3, size=120)
y = 2 + 3 * x - x**2 + rng.normal(scale=0.5, size=120)

# Pick the degree on a validation split
best_degree, best_mse, results = select_degree(
    x[:84], y[:84], x[84:102], y[84:102], degrees=range(1, 6), alpha=1.0
)

model = PolynomialRegression(degree=best_degree, alpha=1.0).fit(x[:84], y[:84])
print(model.coef_)          # e.g. [2.0, 3.0, -1.0]
print(model.predict([0.0, 1.0]))
print(model.score(x[102:], y[102:]))  # R^2
```

`X` may be shape `(n,)` or `(n, 1)`; `predict` returns shape `(n,)`.

## Run

```bash
python polynomial_regression/demo.py
```

## Tests

```bash
python -m pytest tests/test_polynomial_regression.py -v
```
