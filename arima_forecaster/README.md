# AR Forecaster

An autoregressive **AR(p)** time-series forecaster fit by ordinary least
squares, using NumPy only — no statsmodels, no SciPy.

The model is

```
y[t] = intercept + phi_1 * y[t-1] + ... + phi_p * y[t-p] + eps[t]
```

with `eps[t]` white noise of variance `sigma2` (estimated from the
in-sample residuals).

## Features

- `ARForecaster(p=2)` — `p` must be a positive integer.
- `fit(series)` — fits coefficients by least squares on the lagged design
  matrix; stores `coef_`, `intercept_`, `sigma2_` and a `fitted` flag.
- `predict(steps=1)` — recursive multi-step point forecast.
- `forecast_interval(steps=1, alpha=0.05)` — returns
  `(mean, lower, upper)`; interval width grows with the horizon using the
  MA (psi) representation of the fitted AR process, so the forecast-error
  variance at horizon `h` is `sigma2 * sum_{j<h} psi_j**2`.

## Usage

```python
import numpy as np
from arima_forecaster.model import ARForecaster

rng = np.random.default_rng(42)
y = np.zeros(200)
for t in range(2, 200):
    y[t] = 0.6 * y[t - 1] - 0.2 * y[t - 2] + rng.normal()

model = ARForecaster(p=2).fit(y)
print(model.predict(steps=5))                       # point forecasts
mean, lower, upper = model.forecast_interval(steps=5)
print(model.coef_, model.intercept_, model.sigma2_)  # fitted parameters
```

## Run the demo

```bash
python arima_forecaster/demo.py
```

## Run the tests

```bash
pytest tests/test_arima_forecaster.py -q
```
