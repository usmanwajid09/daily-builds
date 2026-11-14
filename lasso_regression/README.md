# Lasso regression, from scratch

Lasso (L1-penalised) linear regression — no scikit-learn, just numpy. It
minimises `0.5 * ||y - Xb||^2 + alpha * ||b||_1` by coordinate descent:
each coefficient is updated in turn with the soft-thresholding operator,
which drives unhelpful coefficients to exactly zero (feature selection).
Features are standardised and the target centred before fitting, so the
penalty treats every feature on the same scale; the returned coefficients
are converted back to the original feature scale.

## Why

Ordinary least squares keeps every feature, even pure noise. The L1
penalty is the simplest way to get a sparse model: with a well-chosen
`alpha`, Lasso keeps the signal and zeroes the rest, which makes the model
both more interpretable and less prone to overfitting.

## Usage

```bash
pip install -r requirements.txt
python -m lasso_regression.demo
```

This generates synthetic data where only 2 of 5 features matter, fits
Lasso with alpha=5.0 on 80% of the data, and prints test R^2 plus the
fitted coefficients on the remaining 20%.

## Run tests

```bash
python -m pytest tests/test_lasso_regression.py -v
```
