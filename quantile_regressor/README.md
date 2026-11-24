# Quantile Regressor (from scratch)

A tiny, dependency-light implementation of linear quantile regression
trained with gradient descent on the pinball loss — numpy + stdlib only,
no scikit-learn.

## Why

Ordinary least squares fits the conditional *mean*. Quantile regression
fits a chosen conditional *quantile*: the median (q = 0.5) is robust to
outliers, and the 0.1 / 0.9 quantiles give a prediction interval. Fitting
several quantiles shows how the spread of `y` changes with `x`, which a
single mean line hides.

## Model

For a quantile `tau`, the pinball loss on a residual `u = y - y_pred` is

```
L_tau(u) = u * (tau - 1{u < 0})
```

so under-prediction costs `tau` per unit and over-prediction costs
`1 - tau` per unit. Minimising it recovers the `tau` conditional
quantile of `y` given `x`.

The loss is piecewise linear, so its subgradient is the step function
`tau - 1{y < pred}` per sample. `QuantileRegressor` runs batch gradient
descent on a smoothed version of that subgradient (the indicator is
softened with a sigmoid of bandwidth 5% of `y`'s standard deviation),
with features standardised internally for stable optimisation. The
coefficients are un-standardised at the end, so `coef_` / `intercept_`
live in the original feature space.

## Usage

```python
from quantile_regressor.model import QuantileRegressor, pinball_loss

# Median regression — robust to outliers
median = QuantileRegressor(quantile=0.5, random_state=0).fit(X, y)
print(median.coef_, median.intercept_, median.n_iter_)

# A 90% prediction interval from the 0.1 / 0.9 quantiles
lo = QuantileRegressor(quantile=0.1, random_state=0).fit(X, y)
hi = QuantileRegressor(quantile=0.9, random_state=0).fit(X, y)

print(median.predict(X))
print(median.score(X, y))                    # R^2, for comparability
print(pinball_loss(y, median.predict(X), 0.5))
```

Parameters: `quantile` (float in (0, 1), default 0.5), `max_iter`
(default 1000), `learning_rate` (default 0.01), `tol` (default 1e-6,
early-stop on pinball-loss change), `fit_intercept` (default True),
`random_state` (optional int, seeds the initial weights).

`fit` validates its inputs and raises `ValueError` for non-2D or
non-finite `X`, mismatched lengths, empty input, or a quantile outside
(0, 1); `predict`/`score` before fitting also raise `ValueError`.

## Demo

```bash
python quantile_regressor/demo.py
```

Generates heteroscedastic data `y = 2x + noise` (noise spread grows
with `x`, seed 0) and fits quantiles 0.1, 0.5 and 0.9. Output:

```
n = 400, x in [0.00, 9.97]
q=0.1: slope=1.617, intercept=-0.513, iterations=3141
q=0.5: slope=1.994, intercept=0.002, iterations=997
q=0.9: slope=2.327, intercept=0.638, iterations=1068
Coverage of q=0.9 model: 0.917 (fraction of y <= prediction)
Pinball loss of median model: 0.724
Median model R^2: 0.896
All demo asserts passed.
```

The quantile lines fan out (slopes 1.62 / 1.99 / 2.33) because the noise
spread grows with `x`; the median recovers the true slope of 2, and the
0.9 line sits above ~92% of the points, close to its nominal 90%.

## Run tests

```bash
python -m pytest tests/test_quantile_regressor.py -q
```

21 tests: pinball-loss hand-computed values, non-negativity and the
zero-loss case, bad-quantile rejection, median slope recovery on clean
data, median robustness to gross outliers, q=0.9 coverage, quantile
ordering (0.9 >= 0.1 predictions), predict shape, R^2 scoring,
determinism with the same `random_state`, no-intercept behaviour, and
`ValueError`s for quantile outside (0, 1), 1D `X`, mismatched lengths,
empty input, NaN, predict-before-fit, and non-positive learning rate.
