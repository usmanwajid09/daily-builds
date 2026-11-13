# K-means clustering, from scratch

A tiny K-means clustering implementation — no scikit-learn, just numpy.
Centroids are initialised with k-means++ and refined with Lloyd's
algorithm until assignments stop changing or `max_iter` is reached. It
runs `n_init` times with different seeds and keeps the run with the
lowest inertia (the sum of squared distances to the assigned centroid).

## Why

K-means is the simplest way to find groups in unlabelled data. Building it
by hand shows why initialisation matters (k-means++ avoids bad local
minima) and what inertia actually measures.

## Usage

```bash
pip install -r requirements.txt
python -m kmeans_clustering.demo
```

This generates three synthetic 2D blobs, clusters them into 3 groups, and
prints the inertia and cluster sizes.

## Run tests

```bash
python -m pytest tests/test_kmeans_clustering.py -v
```
