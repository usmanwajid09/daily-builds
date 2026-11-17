"""Demo: agglomerative clustering on 3 well-separated Gaussian blobs.

Generates 3 blobs (20 points each, seed 42), fits with n_clusters=3 and
linkage="complete", prints cluster sizes and the number of merges, and
asserts the recovered clusters match the true blobs exactly.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agglomerative_clustering.model import AgglomerativeClustering
import numpy as np


def main():
    rng = np.random.default_rng(42)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    X = np.vstack([rng.normal(loc=c, scale=0.5, size=(20, 2)) for c in centers])
    y_true = np.repeat([0, 1, 2], 20)

    model = AgglomerativeClustering(n_clusters=3, linkage="complete")
    labels = model.fit_predict(X)

    sizes = [int((labels == k).sum()) for k in range(3)]
    print("Cluster sizes:", sizes)
    print("Number of merges:", len(model.merges_))

    assert sorted(sizes) == [20, 20, 20], f"expected three clusters of 20, got {sizes}"
    # Each true blob must map to a single predicted label.
    for blob in range(3):
        blob_labels = set(labels[y_true == blob].tolist())
        assert len(blob_labels) == 1, (
            f"true blob {blob} split across labels {blob_labels}"
        )
    print("Demo OK: each true blob recovered as a single cluster.")


if __name__ == "__main__":
    main()
