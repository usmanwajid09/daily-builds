# Spectral Clustering, from scratch

Spectral clustering, built with numpy only — no scikit-learn. An RBF
affinity graph is built over the data, the symmetric normalized graph
Laplacian is eigendecomposed, and the data is embedded in the space of
its smallest eigenvectors, where k-means (k-means++ init) finds the
clusters. Non-convex shapes such as concentric rings or moons, which
plain k-means cannot split, become separable in the spectral embedding.

## Usage

```bash
pip install -r requirements.txt
python spectral_clustering/demo.py
```

This clusters two synthetic concentric rings and prints the cluster
sizes, the smallest Laplacian eigenvalues, and the accuracy against the
true ring labels.

## Run tests

```bash
python -m pytest tests/test_spectral_clustering.py -v
```
