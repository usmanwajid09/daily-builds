# KNN Regressor

K-nearest neighbours regression, from scratch with numpy only. No
scikit-learn, no training loop — the model just stores the training
data and, for each query point, averages the targets of its `k`
nearest neighbours (Euclidean distance).

## Why KNN regression?

It is the simplest non-parametric regressor: it makes no assumption
about the shape of the underlying function, so it can fit curves like
`sin(x)` that a straight line cannot, and it works out of the box on
any number of features. The trade-off is that it stores the whole
training set and gets slow (and less reliable) as the data grows or
the feature space gets high-dimensional.

## Uniform vs distance weighting

- **uniform** (default): the prediction is the plain mean of the `k`
  nearest targets. Every neighbour counts equally, whatever its
  distance.
- **distance**: each neighbour is weighted by `1 / distance`, so
  nearer points pull the prediction more strongly. This usually
  follows local structure better and reduces the smoothing effect of
  far-away neighbours. A query point that exactly matches a training
  point (distance 0) returns the mean of the matching targets.

## Usage

```python
import numpy as np
from knn_regressor import KNNRegressor

x = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([0.0, 1.0, 4.0, 9.0])

model = KNNRegressor(k=3, weights="distance").fit(x, y)
print(model.predict([[1.5]]))   # weighted mean of the 3 nearest targets
print(model.score(x, y))        # R^2 on the training data
```

## Run the demo

```bash
python knn_regressor/demo.py
```

Fits both weightings (`k=5`) to 100 noisy `sin(x)` points and reports
test MSE and R^2 on 50 held-out points, plus one sample prediction
against its true value.

## Run the tests

```bash
python -m pytest tests/test_knn_regressor.py -q
```
