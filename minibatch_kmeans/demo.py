"""Generates three synthetic 2D clusters, fits MiniBatch K-Means, reports inertia and cluster purity."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minibatch_kmeans.model import MiniBatchKMeans


def make_blobs(seed: int = 42):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.5]])
    x = np.vstack([rng.normal(loc=c, scale=0.8, size=(100, 2)) for c in centres])
    y = np.array([0] * 100 + [1] * 100 + [2] * 100)
    perm = rng.permutation(len(y))
    return x[perm], y[perm], centres


def purity(labels, y_true, n_clusters):
    correct = 0
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            continue
        values, counts = np.unique(y_true[mask], return_counts=True)
        correct += counts.max()
    return correct / len(y_true)


def main():
    x, y, true_centres = make_blobs()
    model = MiniBatchKMeans(n_clusters=3, batch_size=50, max_iter=100,
                            random_state=42)
    model.fit(x)

    labels = model.labels_
    print(f"Inertia:        {model.inertia_:.2f}")
    print(f"Iterations run: {model.n_iter_}")
    print(f"Cluster purity: {purity(labels, y, 3):.3f}")
    print("Found centres:")
    for centre in np.round(model.cluster_centers_, 2):
        print(f"  {centre.tolist()}")
    print("True centres:")
    for centre in true_centres:
        print(f"  {centre.tolist()}")
    print("Cluster sizes:", np.bincount(labels).tolist())


if __name__ == "__main__":
    main()
