# Agglomerative Clustering (from scratch)

Hierarchical agglomerative clustering, implemented from scratch with
numpy only. Every point starts in its own cluster; the two closest
clusters are merged repeatedly until `n_clusters` remain.

## Linkage rules

Cluster distance is recomputed from member points at each merge:

- **single** — distance between the closest pair of points across the
  two clusters (can chain clusters together).
- **complete** — distance between the farthest pair of points (favours
  compact, roughly equal-sized clusters).
- **average** — mean distance over all cross pairs of points (a middle
  ground between single and complete).

## Usage

```python
from agglomerative_clustering.model import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3, linkage="complete")
labels = model.fit_predict(X)
```

After `fit`, the model exposes `labels_` (0 .. n_clusters-1), `merges_`
(a list of `(cluster_a, cluster_b, distance, size)` per merge),
`children_` (merge pairs), and `n_clusters_`.

## Run the demo / tests

```bash
python agglomerative_clustering/demo.py
python -m pytest tests/test_agglomerative_clustering.py -q
```

## Demo output

3 Gaussian blobs, 20 points each (seed 42), `n_clusters=3`,
`linkage="complete"`:

```text
Cluster sizes: [20, 20, 20]
Number of merges: 57
Demo OK: each true blob recovered as a single cluster.
```

60 points start as 60 singleton clusters; 57 merges leave 3 clusters,
each of exactly 20 points, matching the true blobs one-to-one.
