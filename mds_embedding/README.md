# Classical MDS Embedding

Classical (metric) multidimensional scaling from scratch (numpy only).
From a square pairwise-distance matrix it double-centres the squared
distances, B = −½·J·D²·J, takes the eigendecomposition of B, and keeps
the top-k positive eigenvalues and eigenvectors as coordinates.
Kruskal's stress-1 measures how well the embedding reproduces the
input distances.

## Run

```bash
python mds_embedding/demo.py
python -m pytest tests/test_mds_embedding.py -q
```

## Example

Demo on 30 points from three 2D clusters (only distances given to MDS):

```
Points: 30, embedding shape: (30, 2)
Eigenvalues: [211.304, 172.19]
Kruskal stress-1: 0.0000
Procrustes correlation with true layout: 1.0000
```

The layout is recovered almost exactly — stress near zero and a
near-perfect correlation after optimal rotation.

## API

- `ClassicalMDS(n_components=2)` — `fit(d)`, `fit_transform(d)`,
  `ClassicalMDS.from_points(x, n_components=2)`
- `.embedding_`, `.eigenvalues_`, `.stress_`
- `pairwise_distances(x)` → Euclidean distance matrix
- `double_centre(d)` → B = −½·J·D²·J
- `kruskal_stress(d, embedding)` → stress-1 in [0, 1]
