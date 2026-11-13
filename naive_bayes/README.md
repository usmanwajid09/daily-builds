# Gaussian Naive Bayes, from scratch

A tiny Gaussian Naive Bayes classifier — no scikit-learn, just numpy.
Each feature is modelled per class as an independent Gaussian, and the
predicted class is the one with the highest log-posterior (log-prior plus
the sum of per-feature log-likelihoods). A small variance floor keeps the
Gaussian well-defined when a feature is constant inside a class.

## Why

Naive Bayes is the simplest probabilistic classifier: training is just
computing a mean and variance per feature per class, yet it works
surprisingly well on text and sensor data, and it gives real
probabilities out of the box.

## Usage

```bash
pip install -r requirements.txt
python -m naive_bayes.demo
```

This generates three synthetic Gaussian classes, fits the classifier on
80% of the data, and prints test accuracy on the remaining 20%.

## Run tests

```bash
python -m pytest tests/test_naive_bayes.py -v
```
