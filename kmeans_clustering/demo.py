"""Generates three synthetic blobs, runs K-means, reports inertia and sizes."""
import numpy as np

from kmeans_clustering.model import KMeans


def make_blobs(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 50
    blobs = [
        rng.normal(loc=[0.0, 0.0], scale=0.8, size=(n, 2)),
        rng.normal(loc=[6.0, 0.0], scale=0.8, size=(n, 2)),
        rng.normal(loc=[3.0, 5.0], scale=0.8, size=(n, 2)),
    ]
    x = np.vstack(blobs)
    perm = rng.permutation(len(x))
    return x[perm]


def main():
    x = make_blobs()
    model = KMeans(n_clusters=3, random_state=42).fit(x)
    sizes = [int(np.sum(model.labels_ == k)) for k in range(3)]

    print(f"Samples: {len(x)}, clusters: 3")
    print(f"Inertia: {model.inertia_:.2f}")
    print(f"Cluster sizes: {sizes}")
    print(f"Iterations: {model.n_iter_}")
    print(f"Centers:\n{np.round(model.cluster_centers_, 2)}")


if __name__ == "__main__":
    main()
