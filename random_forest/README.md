# Random-forest classifier, from scratch

A random-forest classifier — no scikit-learn, just numpy. Each tree is a
small CART classifier (Gini impurity, greedy splits) grown on a bootstrap
resample of the data, and at every split only a random subset of the
features is considered. Predictions are by majority vote across the trees,
and `predict_proba` averages the trees' leaf probabilities.

## Why

A single decision tree has high variance: a small change in the data can
change the whole tree. Bagging many trees on resampled data and decorrelating
them with random feature subsets averages that variance away, which is why
random forests are a strong default classifier. Building one by hand shows
exactly where the variance reduction comes from.

## Usage

```bash
pip install -r requirements.txt
python -m random_forest.demo
```

This generates three synthetic 2D clusters, fits a forest of 50 trees with
max_depth=6 on 80% of the data, and prints test accuracy on the remaining
20%.

## Run tests

```bash
python -m pytest tests/test_random_forest.py -v
```
