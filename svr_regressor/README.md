# SVR Regressor — Support Vector Regression from Scratch

A linear Support Vector Regression (SVR) model implemented with NumPy only,
trained by stochastic gradient descent (SGD) on the primal objective. No
scikit-learn, no new dependencies.

## What it does

`SVRRegressor` fits a linear model `y = w·x + b` that is deliberately
insensitive to small errors. It standardises the features internally,
shuffles the data each epoch (seeded, so runs are reproducible), and
updates the weights one sample at a time.

```python
from svr_regressor.model import SVRRegressor

model = SVRRegressor(epsilon=0.05, C=10.0, learning_rate=0.01,
                     n_epochs=500, random_state=42)
model.fit(X_train, y_train)
print(model.predict(X_test))
print(model.score(X_test, y_test))  # R²
```

Constructor parameters: `epsilon=0.1` (tube half-width), `C=1.0`
(penalty for errors outside the tube), `learning_rate=0.01`,
`n_epochs=500`, `random_state=42`. After fitting, the model exposes
`coef_`, `intercept_`, and `n_features_in_`. `fit` validates its inputs
and raises `ValueError` for 1D `X`, mismatched lengths, empty input,
NaN/inf values, `epsilon < 0`, `C <= 0`, `learning_rate <= 0`, or
`n_epochs < 1`.

## The epsilon-insensitive loss idea

Ordinary least squares punishes every error, however tiny. SVR instead
minimises

    0.5 * ||w||² + C * Σ max(0, |yᵢ − (w·xᵢ + b)| − ε)

Errors smaller than `ε` fall inside a "tube" around the prediction and
cost nothing — only points outside the tube pull on the weights. The
result is a model that ignores small noise and focuses on the points
that actually disagree with it (the support vectors), while the `0.5·||w||²`
term keeps the weights small and the fit smooth. Larger `C` fits the data
harder; larger `ε` tolerates more noise.

## How to run

From the repo root:

```bash
python svr_regressor/demo.py
python -m pytest tests/test_svr_regressor.py -q
```

## Demo output

The demo fits a noisy sine wave (`sin(x)` on [0, 1], Gaussian noise
σ = 0.05, seed 42, n = 120, 80/20 split):

```text
Train size: 96
Test size: 24
Test R^2: 0.9818
Test MSE: 0.0011
```

## Tests

12 pytest tests in `tests/test_svr_regressor.py`: fits a noiseless linear
target (R² > 0.9) and a sine wave (R² > 0.7), predict shape, `score`
returns a float in [-1, 1], fitted attributes, `ValueError` for 1D `X`,
mismatched lengths, empty input, NaN input, negative `C`, and negative
`epsilon`, plus determinism with a fixed seed. All 12 pass.
