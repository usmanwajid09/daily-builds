# Gradient-boosting regressor, from scratch

A gradient-boosting regressor — no scikit-learn, just numpy. It starts
from the mean of `y` and, at each stage, fits a shallow regression tree to
the current residuals (the negative gradient of the squared loss) and adds
a shrunken copy of it to the prediction. Shrinkage (`learning_rate`) plus
shallow trees is what keeps the model from memorising the training data.

## Why

Boosting turns weak learners into a strong one by fixing mistakes
sequentially: each new tree is trained on what the previous ones got
wrong. It is the idea behind XGBoost and LightGBM, the workhorses of
tabular machine learning. Building it by hand shows how residuals, depth,
and learning rate interact.

## Usage

```bash
pip install -r requirements.txt
python -m gradient_boosting.demo
```

This generates synthetic regression data with a non-linear target, fits
150 depth-3 trees on 80% of the data, and prints test R^2 and RMSE on the
remaining 20%.

## Run tests

```bash
python -m pytest tests/test_gradient_boosting.py -v
```
