# PLS Regression (NIPALS), from scratch

Partial least squares regression implemented with the NIPALS
algorithm, using numpy only — no scikit-learn. PLS extracts latent
components that maximise the covariance between the predictors `X`
and the responses `y`, which makes it stable when the features are
strongly collinear — exactly where ordinary least squares becomes
unstable. Multi-output `y` (PLS2) is supported, as is 1D `y`.

The data are always centred; with `scale=True` (the default) `X`
and `y` are additionally standardised to unit variance. A constant
(zero-variance) column gets a scale factor of 1 instead of dividing
by zero, so it is left alone rather than crashing the fit.

## Usage

```bash
pip install -r requirements.txt
python -m pls_regression.demo
```

This generates 100 samples with 10 features built from 3 latent
factors (highly collinear), fits PLS with 1, 2 and 3 components and
prints the training R^2 for each, plus the coefficients of the
3-component model.

## Example output

```
Data: 100 samples, 10 features (built from 3 latent factors, so highly collinear)
n_components=1: train R^2 = 0.6xxx
n_components=2: train R^2 = 0.8xxx
n_components=3: train R^2 = 0.9xxx
coef_ shape: (10, 1)
First 5 coefficients: [...]
intercept: x.xxx
X-scores shape (transform): (100, 3)
```

(The exact R^2 values depend on the random draw; with 3 components
the model recovers most of the signal because the data really have
3 latent factors.)

## API

```python
from pls_regression import PLSRegression

model = PLSRegression(n_components=2, scale=True)
model.fit(X, y)
model.predict(X)    # 1D array for 1D y, 2D for multi-output y
model.score(X, y)   # R^2 (mean over targets for multi-output)
model.transform(X)  # X-scores, shape (n_samples, n_components)
```

Attributes after fitting: `coef_` (shape `(n_features, n_targets)`),
`intercept_`, `x_weights_`, `x_loadings_`, `y_loadings_`,
`x_scores_`, `y_scores_`, `x_rotations_` (`W (P^T W)^-1`).

## Run tests

```bash
python -m pytest tests/test_pls_regression.py -q
```
