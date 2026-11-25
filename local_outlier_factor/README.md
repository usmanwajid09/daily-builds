# Local Outlier Factor (LOF)

Anomaly detection from scratch with numpy only — no scikit-learn, no scipy.

## What it does

`LocalOutlierFactor` scores each point by how isolated it is compared with
its neighbours. A point deep inside a dense cluster gets a score near 1;
a point far from everyone gets a much larger score. It flags the
`contamination` fraction of points with the worst scores as outliers
(label `-1`, inliers are `1`).

## How it works

For each point `p` (Breunig et al., 2000):

1. **k-distance(p)** — distance to its k-th nearest neighbour.
2. **Reachability distance** `reach(p, o) = max(k-distance(o), dist(p, o))`
   — smooths the density estimate.
3. **Local reachability density** `lrd(p) = 1 / mean reachability distance
   to its k neighbours`.
4. **LOF(p)** — mean of `lrd(neighbour) / lrd(p)` over the k neighbours.
   LOF ≈ 1 means "as dense as my neighbours"; LOF >> 1 means "outlier".

Scores follow the scikit-learn convention: `score_samples` /
`decision_function` return the **negative** LOF, so higher = more normal
and outliers score negative. `predict` compares the score against a
threshold set from `contamination`.

## Usage

```python
from local_outlier_factor.model import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
labels = lof.fit_predict(X)        # 1 = inlier, -1 = outlier
scores = lof.score_samples(X)      # negative LOF, higher = more normal
print(lof.negative_outlier_factor_)  # per-training-point scores
```

## Run

```bash
python local_outlier_factor/demo.py
pytest tests/test_local_outlier_factor.py -v
```
