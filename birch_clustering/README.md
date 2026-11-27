# BIRCH Clustering

BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
from scratch, using only numpy.

## What it does

`BIRCH` clusters data in two stages. First it compresses the points into
a small set of **leaf subclusters** stored in a CF-tree. Then it groups
those subclusters into the requested number of final clusters with a
deterministic k-means. This keeps the memory footprint small: the tree
stores summaries, not the raw points.

## How it works

**Clustering features.** Each subcluster is summarised by a triple
`(N, LS, SS)`: the point count `N`, the vector sum `LS` of its points,
and the sum of squared norms `SS`. From a CF alone we recover the
centroid (`LS / N`) and the radius
(`sqrt(SS / N - ||centroid||**2)`, clamped at 0).

**CF-tree.** Points are inserted one by one. A point joins the closest
leaf entry if the merged entry's radius stays within `threshold`;
otherwise it starts a new entry. A leaf holding more than
`branching_factor` entries splits: the two farthest entries become
seeds and the rest join the nearer seed. Internal nodes split the same
way when they exceed `branching_factor` children, growing the tree
upwards from the root.

**Final step.** The leaf subcluster centroids (weighted by their point
counts) are clustered with a small in-file k-means — k-means++ style
initialisation with a fixed seed, so results are repeatable. If
`n_clusters` is `None`, each leaf subcluster is its own final cluster.
If there are fewer leaf subclusters than `n_clusters`, all of them are
used.

## Parameters

- `threshold` (default `0.5`): max radius of a leaf subcluster.
  Smaller values give more, finer subclusters.
- `branching_factor` (default `50`): max entries per leaf / children
  per internal node before a split.
- `n_clusters` (default `3`): number of final clusters, or `None` to
  keep every leaf subcluster.

## Usage

```python
from birch_clustering.model import BIRCH

model = BIRCH(n_clusters=3)
labels = model.fit_predict(X)

model.labels_           # cluster label per point
model.cluster_centers_  # final cluster centres
model.n_clusters_       # clusters actually produced
model.n_subclusters_    # leaf subclusters in the CF-tree
model.subclusters_      # leaf subcluster centroids
```

Invalid inputs raise `ValueError`: non-2D or empty `X`, non-finite
values, `threshold <= 0`, `branching_factor < 2`, or `n_clusters < 1`.

## How to run

```bash
cd /home/hatch/workspace/daily-builds
python birch_clustering/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_birch_clustering.py -q
```

## Example output

```
Leaf subclusters: 9
Clusters: 3
Cluster sizes: [30, 30, 30]
Cluster centres:
 [[ 4.99  9.96]
 [ 0.06  0.  ]
 [ 9.91 -0.09]]
OK: each true blob recovered as a single cluster.
```
