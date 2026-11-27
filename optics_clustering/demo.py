"""Demo: OPTICS on two Gaussian blobs plus scattered noise points.

Run from the repo root:  python optics_clustering/demo.py
"""
import sys
from pathlib import Path

# Running a script inside its package folder does not put the repo root
# on sys.path, so add it for the package import below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from optics_clustering.model import OPTICS

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

model = OPTICS(min_samples=5, xi=0.05).fit(x)
labels = model.labels_

print(f"Points: {len(x)} (120 in blobs, 8 planted noise)")
print(f"Clusters found: {model.n_clusters_}")
for k in range(model.n_clusters_):
    print(f"  cluster {k}: {int(np.sum(labels == k))} points")
n_noise = int(np.sum(labels == -1))
print(f"Noise points: {n_noise}")
planted_noise_labelled_noise = int(np.sum(labels[120:] == -1))
print(f"Planted noise points labelled noise: {planted_noise_labelled_noise} / 8")
ordering, reachability = model.reachability_plot_data()
print("First 10 reachability distances in OPTICS order:")
print(" ", [f"{v:.3f}" if np.isfinite(v) else "inf" for v in reachability[:10]])
print(f"Ordering starts at point {ordering[0]} "
      f"(reachability inf: reached from nowhere)")
