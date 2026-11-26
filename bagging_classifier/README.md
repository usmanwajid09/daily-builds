# Bagging classifier, from scratch

A bootstrap-aggregated decision-tree classifier — no scikit-learn, just
numpy. Each base learner is a small CART tree (Gini splits) trained on a
bootstrap resample of the data; predictions are by majority vote. Bagging
averages out the high variance of individual trees.

## Why

A single deep tree overfits its training data. Training many trees on
different resamples and voting smooths away that variance, which is the
core idea behind random forests (this version uses all features at every
split unless `max_features` is set).

## Usage

```bash
pip install -r requirements.txt
python bagging_classifier/demo.py
```

This fits one tree and a 25-tree bagged ensemble on two-moons data and
compares test accuracy.

## Demo output (seed 42)

```
Train size: 160, test size: 40
Single tree accuracy:   0.875
Bagged (25 trees) acc:  0.950
Improvement:            +0.075
First 5 predictions: [0, 1, 0, 0, 1]
First 5 true labels: [0, 1, 1, 0, 1]
```

## Run tests

```bash
python -m pytest tests/test_bagging_classifier.py -v
```
