# Gaussian mixture model, from scratch

A Gaussian mixture model with full covariance matrices — no scikit-learn,
just numpy. It is fit by expectation-maximisation: the E-step computes
soft responsibilities for each point under each component, and the M-step
re-estimates the weights, means and covariances from those
responsibilities. A small `reg_covar` is added to every covariance
diagonal so the likelihood stays finite even if a component collapses
onto a few points.

## Why

Mixture models are the probabilistic way to cluster: unlike k-means they
model each cluster's shape (full covariance) and report how confident they
are about every assignment via `predict_proba`. Building EM by hand shows
how a soft clustering converges and why covariance regularisation matters.

## Usage

```bash
pip install -r requirements.txt
python -m gaussian_mixture.demo
```

This generates two synthetic 2D clusters, fits a 2-component mixture, and
prints the recovered means, weights, log-likelihood, and clustering
accuracy against the true blob labels.

## Run tests

```bash
python -m pytest tests/test_gaussian_mixture.py -v
```
