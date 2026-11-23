# Nearest Centroid

Nearest-centroid classifier with optional shrinkage, from scratch with
numpy only.

## Idea

Each class is summarised by its centroid (mean feature vector); a new
sample is assigned to the class whose centroid is nearest in Euclidean
(or Manhattan) distance. Optionally, centroids are shrunk toward the
overall mean by a factor `shrinkage` in [0, 1], which regularises the
classifier when a class has few samples or noisy features.

## Usage

```python
from nearest_centroid.model import NearestCentroid

model = NearestCentroid(shrinkage=0.1).fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)
```

## Parameters

- `shrinkage` (default 0.0): fraction each centroid is pulled toward the
  overall mean; must be in [0, 1].
- `metric` (default `"euclidean"`): `"euclidean"` or `"manhattan"`.

## Demo

```bash
python nearest_centroid/demo.py
```

Fits on three synthetic 2D clusters and prints test accuracy for
shrinkage 0.0, 0.3 and 0.7, plus the Manhattan metric.

## Tests

```bash
python -m pytest tests/test_nearest_centroid.py -q
```
