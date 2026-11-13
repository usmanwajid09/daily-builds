# Decision tree classifier, from scratch

A tiny CART decision-tree classifier — no scikit-learn, just numpy.
Splits are chosen greedily to minimise the weighted Gini impurity of the
child nodes. Growth stops when a node is pure, when `max_depth` is
reached, or when a node has fewer than `min_samples_split` samples.
Leaves predict the majority class, and `predict_proba` returns the leaf's
class frequencies.

## Why

Decision trees are the most interpretable classifier: every prediction can
be traced through a short chain of "feature <= threshold" rules. Building
one by hand shows how Gini impurity drives the splits and why depth limits
matter for overfitting.

## Usage

```bash
pip install -r requirements.txt
python -m decision_tree.demo
```

This generates two synthetic 2D clusters, fits a tree with max_depth=4 on
80% of the data, and prints test accuracy on the remaining 20%.

## Run tests

```bash
python -m pytest tests/test_decision_tree.py -v
```
