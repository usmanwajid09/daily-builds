# OPTICS Clustering

OPTICS (Ordering Points To Identify the Clustering Structure),
implemented from scratch with numpy only.

Unlike k-means, OPTICS needs no `k` up front, finds clusters of
arbitrary shape, and explicitly labels outliers as noise. Unlike
DBSCAN, it does not commit to a single global radius: it builds an
ordering of the points and reads the clusters off the reachability
plot, so it can handle clusters of somewhat different densities.

## How it works

1. **Core distance** — for each point, the distance to its
   `(min_samples - 1)`-th nearest *other* point. A point deep inside
   a dense region has a small core distance; a point in a sparse
   region has a large one (or `inf` if it has fewer than
   `min_samples - 1` neighbours within `max_eps`). With
   `min_samples == 1` every point is trivially core, so the core
   distance is 0.
2. **Reachability distance** — of a point `q` with respect to a point
   `p`: `max(core_distance(p), distance(p, q))`. It is the smallest
   radius around `p` that both makes `p` core and reaches `q`.
3. **Ordering** — start from an arbitrary point, then repeatedly take
   the unprocessed point with the smallest reachability distance
   from any already-processed point (a min-heap of "seeds"). Points
   in the same dense region follow each other with small
   reachability distances, so the ordering walks through one cluster
   at a time. The first point's reachability is `inf` (it is reached
   from nowhere).
4. **Xi cluster extraction** — clusters are valleys in the
   reachability plot. A cluster boundary is a place where the
   reachability rises or falls *steeply*: the extraction looks for
   the first (lowest) multiplicative gap in the sorted finite
   reachability values above the typical (median) level, where the
   gap is at least as steep as `1 / (1 - xi)` and at least a
   doubling, so ordinary within-cluster variation never splits a
   cluster. Points above that level are separators; the entry point
   of a cluster (the separator just before a run of low-reachability
   points) belongs to the cluster it enters. A run only counts as a
   cluster if, entry point included, it holds at least `min_samples`
   points; anything shorter is noise, labelled `-1`. If no steep gap
   exists, all connected points form a single cluster.

   Limitation: the extraction assumes clusters of roughly similar
   density — a much sparser cluster's own reachability level can sit
   above the separation level of a denser one.

## Parameters

- `min_samples` (default 5) — defines the core distance and the
  minimum cluster size. Must be a positive integer.
- `xi` (default 0.05) — steepness threshold for cluster extraction,
  in `(0, 1)`. Smaller `xi` = less steep a jump needs to be to count
  as a boundary.
- `max_eps` (default `None` = infinity) — maximum radius considered
  when looking for neighbours. Must be a positive finite number if
  given.

## Usage

```python
import numpy as np
from optics_clustering.model import OPTICS

x = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0]])
model = OPTICS(min_samples=3).fit(x)
print(model.labels_)        # e.g. [0 0 0 1 1]
print(model.n_clusters_)    # 2
print(model.ordering_)      # point indices in OPTICS order
print(model.reachability_)  # reachability per point in that order
print(model.core_distances_)  # core distance per point (original order)
ordering, reachability = model.reachability_plot_data()
```

Noise points get label `-1`; cluster labels run `0..n_clusters-1`.
`n_clusters_` counts clusters only, not noise. Fitting is
deterministic — the same data always gives the same labels.
`cluster_hierarchy_` lists one entry per cluster with its `label`,
`start`/`end` positions in the ordering, and `size`.

## Run the demo

From the repo root:

```bash
python optics_clustering/demo.py
```

Two Gaussian blobs (60 points each) plus 8 scattered noise points,
fit with `min_samples=5, xi=0.05`.

Example output:

```
Points: 128 (120 in blobs, 8 planted noise)
Clusters found: 2
  cluster 0: 60 points
  cluster 1: 60 points
Noise points: 8
Planted noise points labelled noise: 8 / 8
First 10 reachability distances in OPTICS order:
  ['inf', '0.112', '0.112', '0.073', '0.073', '0.073', '0.112', '0.087', '0.087', '0.087']
Ordering starts at point 0 (reachability inf: reached from nowhere)
```

## Run the tests

From the repo root:

```bash
python -m pytest tests/test_optics_clustering.py -q
```

18 tests with fixed seeds: two/three well-separated blobs, planted
and far noise labelled `-1`, ordering as a permutation,
reachability length and first entry `inf`, non-negative and known
core distances, a single blob giving one cluster, `min_samples` /
`xi` / `max_eps` / `X` validation errors, `fit` returning `self`,
labels consistent with `n_clusters_`, `reachability_plot_data()`,
tiny `max_eps` leaving everything noise, determinism, and duplicate
points clustering together.
