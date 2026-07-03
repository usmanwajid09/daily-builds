# daily-builds

Daily small ML/CV/web-dev projects, built and reviewed automatically each day. Each project lives in its own folder at the repo root (see below for today's).

---

# Linear Regression via Gradient Descent (from scratch)

A tiny, dependency-light implementation of simple linear regression trained
with batch gradient descent — no scikit-learn, just numpy.

## Why

Most people use `sklearn.linear_model.LinearRegression` and never see what's
happening underneath. This project implements the actual gradient descent
loop so the mechanics (learning rate, convergence, loss curve) are visible
and tweakable.

## Usage

```bash
pip install -r requirements.txt
python -m linreg_gd.demo
```

This generates a synthetic noisy linear dataset, fits it, prints the learned
slope/intercept vs the true ones, and saves `fit_plot.png` showing the data
and the fitted line.

## Run tests

```bash
python -m pytest tests/
```
