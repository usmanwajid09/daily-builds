# Histogram-Based Gradient Boosting Classifier

A gradient boosting classifier with histogram-binned splits, built from scratch with numpy only.

## What it does

Features are first quantised into `max_bins` quantile bins; each
boosting round then fits a small regression tree to the log-loss
gradients and Hessians using only bin-level gradient sums, which keeps
the split search fast. Leaf values use the Newton step
`-sum(grad) / (sum(hess) + lambda)`. Binary classification optimises
logistic loss directly; multiclass problems use a one-vs-rest ensemble
of boosters.

## How to run

```bash
python histogram_gb_classifier/demo.py
```

```python
from histogram_gb_classifier.model import HistogramGBClassifier

model = HistogramGBClassifier(n_estimators=80, learning_rate=0.1,
                              max_depth=3, max_bins=16)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

## Demo result

Two interleaving moons (150 points per class, 80/20 train/test
split), 80 trees, learning rate 0.1, depth 3, 16 bins.

- Train accuracy: **1.000**
- Test accuracy: **0.967**

## Example output

```
Train accuracy: 1.000
Test accuracy:  0.967
Trees per class: 80
Bins per feature: 16
First 5 test predictions: [0, 1, 1, 0, 0]
First 5 true labels:      [0, 1, 1, 0, 0]
First test point probabilities: [0.889, 0.111]
```

## Tests

10 pytest tests in `tests/test_histogram_gb_classifier.py` —
accuracy on blobs and on non-linear moons, probabilities summing to
1, multiclass one-vs-rest, more estimators not hurting the training
fit, a constant feature not crashing, and input validation.
