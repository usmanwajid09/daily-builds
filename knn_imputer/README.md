# kNN Imputer

k-nearest-neighbour missing-value imputation, implemented from scratch with
numpy only — no scikit-learn.

## How it works

For each row with missing entries, the `k` nearest donor rows are found using
Euclidean distance computed over the features observed in **both** rows, with
the squared distance scaled by `n_features / n_common` so rows sharing fewer
features stay comparable. Each missing entry is filled with the mean of the
donors that actually observed that feature — a plain mean for
`weights="uniform"`, or an inverse-distance-weighted mean for
`weights="distance"` (a donor at distance 0 dominates). If no donor observed
the feature, the column mean (from fit time) is used; a row with no observed
features at all is filled with column means. A column that is entirely
missing raises `ValueError` at fit time.

## Usage

```python
import numpy as np
from knn_imputer import KNNImputer

X = np.array([[1.0, np.nan], [1.0, 2.0], [3.0, 4.0]])
filled = KNNImputer(n_neighbors=2).fit_transform(X)
```

## Run the demo

```bash
python knn_imputer/demo.py
```

## Run the tests

```bash
python -m pytest tests/test_knn_imputer.py -q
```

## Example output

```
weights=uniform: imputed 55 entries, RMSE on masked entries = 0.7246
weights=distance: imputed 55 entries, RMSE on masked entries = 0.6886

Before / after (first 4 rows with missing values, NaN = masked):
row  1 masked: [    nan  -2.180   1.094  -0.410]
row  1 filled: [ -0.877  -2.180   1.094  -0.410]
...
NaNs remaining after imputation: 0
```
