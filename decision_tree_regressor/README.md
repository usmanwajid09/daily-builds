# Decision Tree Regressor (CART, from scratch)

A regression tree built from scratch with numpy only — no scikit-learn,
no new dependencies.

## What it does

`DecisionTreeRegressor` fits a CART regression tree. It recursively splits
the training data until a stopping rule fires (`max_depth`,
`min_samples_split`, `min_samples_leaf`, or a pure node), and each leaf
predicts the mean of the training targets that reach it.

```python
from decision_tree_regressor.model import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X_train, y_train)
preds = tree.predict(X_test)
print(tree.score(X_test, y_test))  # R²
```

Attributes after fitting: `tree_` (nested dicts), `n_features_in_`,
`max_depth_` (actual depth reached), `n_leaves_`.

## The split idea: variance reduction

At each node the tree tries every feature and every candidate threshold —
the midpoints between consecutive sorted unique values of that feature.
For each candidate it computes the weighted MSE of the two child groups
and compares it with the parent node's MSE:

```
gain = MSE(parent) − (n_left · MSE(left) + n_right · MSE(right)) / n_parent
```

The split with the largest gain (largest variance reduction) wins.
A split is rejected if either child would hold fewer than
`min_samples_leaf` samples. Splitting stops when no split reduces the
variance, or a stopping rule fires.

## How to run

From the repo root:

```bash
python decision_tree_regressor/demo.py
python -m pytest tests/test_decision_tree_regressor.py -q
```

## Demo output

The demo fits a tree (`max_depth=5`) to a noisy sine wave
(seed 42, n=150, 80/20 split) and reports:

```
Test R²: 0.9291
Test MSE: 0.0208
Tree depth: 5
Leaf count: 28
Demo passed: test R² > 0.85
```

## Tests

12 pytest tests cover: near-exact fit on a 1D step function (R² > 0.99),
sine-wave fit (R² > 0.8), `max_depth=1` giving exactly 2 leaves,
`min_samples_leaf` respected, constant `y` predicting the constant,
predict shape, and `ValueError`s for 1D `X`, mismatched lengths, empty
input, NaN input, `max_depth=0`, and `min_samples_leaf=0`. All 12 pass.
