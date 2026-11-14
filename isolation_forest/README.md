# Isolation Forest

An isolation forest anomaly detector, built from scratch with numpy only.

## Why it works

Instead of modelling what "normal" looks like, an isolation forest models
how *hard* a point is to isolate. Each tree recursively splits the data by
picking a random feature and a random split value between that feature's
min and max in the current node, until every leaf holds one sample (or a
depth limit is reached). Anomalies sit far from the rest of the data, so
they get cut off in very few splits — a short path length. Normal points
are packed together and need many splits to separate — a long path length.
Averaging the path length over a forest of such trees gives a robust
anomaly score:

```
score = 2 ** (-mean_path_length / c(n))
```

where `c(n)` is the average path length of an unsuccessful search in a
binary search tree of `n` nodes. Scores near 1 mean "isolated almost
immediately" (very anomalous); scores near 0.5 mean an average path
length. The decision threshold is the `(1 - contamination)` quantile of
the training scores, so the top `contamination` share of training scores
are labelled outliers.

## Usage

```python
import numpy as np
from isolation_forest import IsolationForest

rng = np.random.default_rng(0)
x = rng.normal(size=(500, 2))

model = IsolationForest(n_estimators=100, contamination=0.05, random_state=0)
model.fit(x)

scores = model.decision_function(x)   # higher = more anomalous, in (0, 1]
labels = model.predict(x)             # 1 = inlier, -1 = outlier
```

## API

- `IsolationForest(n_estimators=100, max_samples=256, contamination=0.1, random_state=None)`
  - `max_samples="auto"` means `min(256, n_samples)`.
- `fit(X)` — builds the forest; sets `threshold_`, `offset_` and `n_features_in_`.
- `decision_function(X)` — anomaly score per sample, in (0, 1].
- `score_samples(X)` — alias for `decision_function`.
- `predict(X)` — 1 for inlier, -1 for outlier.

## Run the demo / tests

From the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 python isolation_forest/demo.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_isolation_forest.py -q
```
