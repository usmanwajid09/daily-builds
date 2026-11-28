# Weighted k-NN Classifier

A k-nearest neighbours classifier with optional distance-weighted voting, built from scratch with numpy only.

## What it does

Each neighbour votes for its class. With `weights="uniform"` every
neighbour counts equally; with `weights="distance"` each vote is
weighted by 1 / (distance + epsilon), so closer neighbours matter
more — this often fixes the boundary errors of plain k-NN when classes
overlap. Supports any Minkowski power `p` (1 = Manhattan,
2 = Euclidean) and multiclass problems, and returns class
probabilities from the weighted votes.

## How to run

```bash
python weighted_knn/demo.py
```

```python
from weighted_knn.model import WeightedKNNClassifier

model = WeightedKNNClassifier(n_neighbors=5, weights="distance")
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.predict_proba(X_test[:1]))
```

## Demo result

Two overlapping synthetic 2D clusters (100 points per class, 80/20
train/test split), k=5.

- Uniform weights: train accuracy 0.981, test accuracy **1.000**
- Distance weights: train accuracy 1.000, test accuracy **1.000**

## Example output

```
weights= 'uniform': train accuracy 0.981, test accuracy 1.000
weights='distance': train accuracy 1.000, test accuracy 1.000
First 5 test predictions: [1, 1, 0, 0, 0]
First 5 true labels:      [1, 1, 0, 0, 0]
First test point probabilities: [0.0, 1.0]
```

## Tests

11 pytest tests in `tests/test_weighted_knn.py` — accuracy for both
weightings, a case where distance weighting beats uniform on overlap,
probabilities summing to 1, Manhattan distance, multiclass, exact
training-point recall with k=1, and input validation.
