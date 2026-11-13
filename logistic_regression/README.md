# Logistic regression, from scratch

A tiny binary logistic regression classifier — no scikit-learn, just
numpy. It trains by batch gradient descent on the log-loss, with an
optional L2 penalty on the weights (the bias is never penalised).
Features are standardised inside `fit`, so no manual scaling is needed.
`predict_proba` returns the probability of each class.

## Why

Logistic regression is the workhorse linear classifier: it turns a linear
score into a calibrated probability through the sigmoid, and the learned
weights say directly how much each feature pushes the prediction.

## Usage

```bash
pip install -r requirements.txt
python -m logistic_regression.demo
```

This generates linearly separable synthetic data, fits logistic regression
on 80% of it, and prints test accuracy on the remaining 20%.

## Run tests

```bash
python -m pytest tests/test_logistic_regression.py -v
```
