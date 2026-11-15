"""Demo: segment a synthetic 3-colour image with k-means."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from kmeans_segmentation.segment import segment_image


def make_image(seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((30, 30, 3))
    img[:15, :15] = [0.9, 0.1, 0.1]   # red block
    img[:15, 15:] = [0.1, 0.9, 0.1]   # green block
    img[15:, :] = [0.1, 0.1, 0.9]     # blue block
    img += rng.normal(scale=0.03, size=img.shape)
    return np.clip(img, 0, 1)


def main():
    img = make_image()
    labels, segmented, centers = segment_image(img, n_clusters=3, seed=0)
    sizes = np.bincount(labels.ravel(), minlength=3)
    print(f"Image shape: {img.shape}, clusters: 3")
    print(f"Cluster sizes (pixels): {sizes.tolist()} (expected ~225, 225, 450)")
    print(f"Cluster centres (RGB):\n{np.round(centers, 2).tolist()}")
    print(f"Unique colours in segmented image: {len(np.unique(segmented.reshape(-1, 3), axis=0))}")
    print(f"Labels shape: {labels.shape}, unique labels: {np.unique(labels).tolist()}")


if __name__ == "__main__":
    main()
