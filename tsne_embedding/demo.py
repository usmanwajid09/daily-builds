"""Demo: t-SNE from scratch on three well-separated Gaussian blobs.

Generates 3 blobs in 5D, embeds them in 2D, and reports the ratio of the
mean intra-cluster distance to the mean inter-cluster distance in the
embedding. A ratio < 1 means points from the same blob stayed together
while different blobs stayed apart.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tsne_embedding import TSNE


def main():
    rng = np.random.default_rng(42)
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [8.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 8.0, 0.0, 0.0, 0.0],
    ])
    X = np.vstack([rng.normal(c, 0.5, size=(30, 5)) for c in centers])
    labels = np.repeat(np.arange(3), 30)

    model = TSNE(perplexity=10, n_iter=500, random_state=0)
    Y = model.fit_transform(X)

    # Mean intra-cluster vs mean inter-cluster pairwise distance in the embedding.
    sq = np.sum(Y * Y, axis=1, keepdims=True)
    D = np.sqrt(np.maximum(sq + sq.T - 2.0 * (Y @ Y.T), 0.0))
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    intra = D[same].mean()
    inter = D[~same].mean()
    ratio = intra / inter

    print(f"Input shape:      {X.shape}")
    print(f"Embedding shape:  {Y.shape}")
    print(f"Mean intra-cluster distance: {intra:.3f}")
    print(f"Mean inter-cluster distance: {inter:.3f}")
    print(f"Intra / inter distance ratio: {ratio:.3f}  (< 1 means clusters preserved)")


if __name__ == "__main__":
    main()
