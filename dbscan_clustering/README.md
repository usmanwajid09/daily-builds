# DBSCAN Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise),
implemented from scratch with numpy only.

## Why DBSCAN

- **Arbitrary shapes:** clusters grow by density reachability, so they can
  be moons, rings, or blobs — not just the round clusters k-means finds.
- **Finds noise:** points in sparse regions are labelled `-1` instead of
  being forced into a cluster.
- **No k needed:** you never say how many clusters there are; the data
  decides, given a density threshold.

## Parameters

- `eps` — maximum distance between two points for them to count as
  neighbours. Smaller `eps` = tighter, more fragmented clusters and more
  noise; larger `eps` = looser, merging clusters.
- `min_samples` — minimum number of neighbours within `eps` (counting the
  point itself) for a point to be a *core point*. Core points seed and
  expand clusters; non-core points within reach of a core point are
  *border points* that join but do not expand a cluster.

## Usage

```python
import numpy as np
from dbscan_clustering.model import DBSCAN

x = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0]])
model = DBSCAN(eps=0.5, min_samples=3).fit(x)
print(model.labels_)        # e.g. [0 0 0 1 1]
print(model.n_clusters_)    # 2
print(model.core_sample_indices_)  # indices of the core points
labels = model.fit_predict(x)      # same as model.labels_
```

Noise points get label `-1`; cluster labels run `0..n_clusters-1`.
`n_clusters_` counts clusters only, not noise. Fitting is deterministic —
the same data always gives the same labels.

## Run the demo

From the repo root:

```bash
python dbscan_clustering/demo.py
```

Two Gaussian blobs (60 points each) plus 8 scattered noise points, fit
with `eps=0.5, min_samples=5`.

## Run the tests

From the repo root:

```bash
python -m pytest tests/test_dbscan_clustering.py -q
```
