# Cross-validator, from scratch

K-fold cross-validation utilities — no scikit-learn, just numpy:

- `KFold` — plain k-fold splitter, shuffled or sequential.
- `StratifiedKFold` — every fold keeps the class proportions of `y`.
- `cross_val_score` — fits a fresh copy of an estimator on each fold and
  returns one score per fold (accuracy for classifiers, R² for
  regressors, or a custom scorer).
- `accuracy_score`, `r2_score` — the two built-in scorers.

## Why

A single train/test split can flatter or punish a model by luck. Averaging
scores over several folds gives a steadier estimate of how the model will
do on unseen data, and stratification keeps small classes represented in
every fold.

## Usage

```bash
pip install -r requirements.txt
python cross_validator/demo.py
```

This runs 5-fold stratified CV of a nearest-centroid classifier on three
synthetic blobs and prints the per-fold and mean accuracy.

## Demo output (seed 42)

```
Samples: 150, classes: 3, folds: 5 (stratified, shuffled)
Fold accuracies: [1.0, 0.967, 1.0, 1.0, 1.0]
Mean accuracy: 0.993 +/- 0.013
```

## Run tests

```bash
python -m pytest tests/test_cross_validator.py -v
```
