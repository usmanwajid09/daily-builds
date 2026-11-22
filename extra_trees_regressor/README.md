# Extra Trees Regressor, from scratch

Extremely randomized trees (Extra-Trees) for regression, built with
numpy only — no scikit-learn. Like a random forest, but each split
threshold is drawn uniformly at random between the feature's min and max
in the node, and the best of a random subset of such candidate splits
is kept. Splits are scored by variance reduction. Full trees are grown
on the whole dataset (or bootstrap samples), and predictions are the
mean of the trees' predictions.

## Usage

```bash
pip install -r requirements.txt
python extra_trees_regressor/demo.py
```

This fits a 100-tree forest on noisy sinusoidal data and prints the
test R^2 of the forest vs. a single extra tree, plus the feature
importances.

## Run tests

```bash
python -m pytest tests/test_extra_trees_regressor.py -v
```

## Demo output

```
Train size: 240, test size: 60
Extra trees (100 trees) test R^2: 0.985
Single extra tree test R^2: 0.950
Feature importances: [0.427, 0.573]
```
