# Extra Trees Classifier, from scratch

Extremely randomized trees (Extra-Trees) for classification, built with
numpy only — no scikit-learn. Like a random forest, but each split
threshold is drawn uniformly at random between the feature's min and max
in the node, and the best of a random subset of such candidate splits is
kept. Full trees are grown on the whole dataset (or bootstrap samples),
and predictions are by majority vote across the forest.

## Usage

```bash
pip install -r requirements.txt
python extra_trees_classifier/demo.py
```

This fits a 100-tree forest on two interleaved synthetic crescents and
prints the test accuracy of the forest vs. a single extra tree, plus the
feature importances.

## Run tests

```bash
python -m pytest tests/test_extra_trees_classifier.py -v
```
