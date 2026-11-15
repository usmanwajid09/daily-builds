# Elastic Net regression, from scratch

Elastic Net (L1 + L2 penalised) linear regression — no scikit-learn,
just numpy. It minimises
`0.5 * ||y - Xb||^2 + alpha * (l1_ratio * ||b||_1 + 0.5 * (1 - l1_ratio) * ||b||_2^2)`
by coordinate descent: each coefficient is updated in turn with the
soft-thresholding operator, with the L2 part shrinking the update
through the denominator. The L1 term selects features (driving some
coefficients to exactly zero) while the L2 term keeps correlated
features together instead of picking one arbitrarily — the best of
Lasso and ridge in one model. Features are standardised and the target
centred before fitting, so the penalty treats every feature on the
same scale; the returned coefficients are converted back to the
original feature scale.

## Why

Pure Lasso picks one feature out of a correlated group and zeroes the
rest; pure ridge keeps everything. Elastic Net is the practical middle
ground: it still does feature selection, but correlated features share
the credit, which is more stable when the data has groups of related
predictors.

## Usage

```bash
pip install -r requirements.txt
python elastic_net/demo.py
```

This generates synthetic data where only 2 of 5 features matter (two
of them correlated), fits Elastic Net with alpha=1.0, l1_ratio=0.5 on
80% of the data, and prints test R^2 and MSE plus the fitted
coefficients on the remaining 20%.

Example result:

```
Test R^2: 0.949
Test MSE: 0.245
Coefficients: [1.83, 0.09, -1.41, 0.0, 0.0]
Intercept: 0.034 (converged in 12 passes)
```

## Run tests

```bash
python -m pytest tests/test_elastic_net.py -v
```
