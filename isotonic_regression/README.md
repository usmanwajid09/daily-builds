# Isotonic Regression

Isotonic regression built from scratch with numpy only, using the pool
adjacent violators algorithm (PAVA).

## What it does

Fits a non-decreasing (or non-increasing) stepwise function to data:
the monotone function closest to the observed `y` in least-squares
terms. Predictions are made by linear interpolation between the
fitted points. Useful for calibrating classifier probabilities and
for any relationship known to be monotone.

Options:

- `increasing=False` fits a non-increasing function.
- `out_of_bounds` controls predictions outside the training range:
  `"nan"` (default), `"clip"`, or `"raise"`.
- `sample_weight` is supported in `fit`.

## How to run

```bash
python isotonic_regression/demo.py
```

## Demo result

Noisy samples from the monotone curve y = sqrt(x) (100 points,
x in [0, 10], noise σ=0.3).

- Fitted values monotone non-decreasing: True
- MSE of noisy y vs true curve: **0.0864**
- MSE of isotonic fit vs true curve: **0.0086**
- R² of isotonic fit: **0.898**
