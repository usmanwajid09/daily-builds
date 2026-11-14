"""Demo: DBSCAN on two Gaussian blobs plus scattered noise points.

Run from the repo root:  python dbscan_clustering/demo.py
"""
import sys
from pathlib import Path

# Running a script inside its package folder does not put the repo root
# on sys.path, so add it for the package import below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from dbscan_clustering.model import DBSCAN

rng = np.random.default_rng(42)

# Two well-separated Gaussian blobs, 60 points each.
blob_a = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(60, 2))
blob_b = rng.normal(loc=[5.0, 5.0], scale=0.3, size=(60, 2))
# 8 scattered noise points, far from both blobs.
noise = np.array([
    [-3.0, -3.0], [-3.0, 3.0], [3.0, -3.0], [8.0, 0.0],
    [0.0, 8.0], [8.0, 8.0], [-2.0, 6.0], [6.0, -2.0],
])
x = np.vstack([blob_a, blob_b, noise])

model = DBSCAN(eps=0.5, min_samples=5).fit(x)
labels = model.labels_

print(f"Points: {len(x)} (120 in blobs, 8 planted noise)")
print(f"Clusters found: {model.n_clusters_}")
for k in range(model.n_clusters_):
    print(f"  cluster {k}: {int(np.sum(labels == k))} points")
n_noise = int(np.sum(labels == -1))
print(f"Noise points: {n_noise}")
planted_noise_labelled_noise = int(np.sum(labels[120:] == -1))
print(f"Planted noise points labelled noise: {planted_noise_labelled_noise} / 8")
print(f"Core points: {len(model.core_sample_indices_)}")
