"""Demo: BIRCH clustering on 3 well-separated Gaussian blobs.

Generates 3 blobs (30 points each, seed 42), fits BIRCH with
n_clusters=3, prints the number of leaf subclusters, the number of
final clusters, the cluster sizes and the cluster centres, and asserts
each true blob is recovered as a single cluster.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from birch_clustering.model import BIRCH
import numpy as np


def main():
    rng = np.random.default_rng(42)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    X = np.vstack([rng.normal(loc=c, scale=0.5, size=(30, 2)) for c in centers])
    y_true = np.repeat([0, 1, 2], 30)

    model = BIRCH(n_clusters=3)
    labels = model.fit_predict(X)

    sizes = [int((labels == k).sum()) for k in range(model.n_clusters_)]
    print("Leaf subclusters:", model.n_subclusters_)
    print("Clusters:", model.n_clusters_)
    print("Cluster sizes:", sizes)
    print("Cluster centres:\n", np.round(model.cluster_centers_, 2))

    assert sorted(sizes) == [30, 30, 30], f"expected three clusters of 30, got {sizes}"
    for blob in range(3):
        blob_labels = set(labels[y_true == blob].tolist())
        assert len(blob_labels) == 1, (
            f"true blob {blob} split across labels {blob_labels}"
        )
    print("OK: each true blob recovered as a single cluster.")


if __name__ == "__main__":
    main()
