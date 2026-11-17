"""Demo: SLIC superpixel segmentation on a synthetic 4-quadrant image.

Builds a small 60x80 RGB image with four differently coloured quadrants
plus a little noise, segments it into 16 superpixels, and prints the
resulting label statistics and boundary-adherence numbers.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from superpixel_segmentation.superpixel_segmentation import slic

import numpy as np


def main():
    rng = np.random.default_rng(0)
    height, width = 60, 80
    image = np.zeros((height, width, 3))
    # Four quadrants: red / green / blue / yellow.
    image[:30, :40] = [0.9, 0.1, 0.1]
    image[:30, 40:] = [0.1, 0.8, 0.1]
    image[30:, :40] = [0.1, 0.1, 0.9]
    image[30:, 40:] = [0.9, 0.9, 0.1]
    image += rng.normal(0, 0.03, image.shape)  # light sensor-like noise
    image = np.clip(image, 0, 1)

    labels, centers = slic(image, n_segments=16, compactness=10.0,
                           max_iter=10, seed=0)

    n_labels = len(np.unique(labels))
    sizes = np.bincount(labels.ravel())
    print(f"Image shape: {image.shape}")
    print(f"Requested segments: 16, produced segments: {n_labels}")
    print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"Segment size: mean={sizes.mean():.1f}, min={sizes.min()}, "
          f"max={sizes.max()} pixels (ideal = {height * width / 16:.0f})")
    print(f"Centers shape: {centers.shape} "
          f"(first center y={centers[0, 0]:.1f}, x={centers[0, 1]:.1f}, "
          f"colour={np.round(centers[0, 2:], 2)})")

    # Boundary adherence: fraction of superpixel borders that sit on a
    # true colour edge (a neighbouring pixel pair with different colours).
    colour_edge = np.zeros((height, width), dtype=bool)
    colour_edge[:, 1:] |= np.abs(image[:, 1:] - image[:, :-1]).sum(axis=2) > 0.5
    colour_edge[1:, :] |= np.abs(image[1:] - image[:-1]).sum(axis=2) > 0.5
    label_edge = np.zeros((height, width), dtype=bool)
    label_edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    label_edge[1:, :] |= labels[1:] != labels[:-1]
    on_true_edge = (label_edge & colour_edge).sum()
    print(f"Superpixel border pixels: {label_edge.sum()}, of which "
          f"{on_true_edge} lie on a true colour edge "
          f"({100 * on_true_edge / max(label_edge.sum(), 1):.1f}%)")

    # Colour purity: mean within-segment colour standard deviation.
    stds = []
    for lab in np.unique(labels):
        stds.append(image[labels == lab].std(axis=0).mean())
    print(f"Mean within-segment colour std: {np.mean(stds):.4f} "
          f"(lower = purer segments)")
    print("First 5 rows of labels:")
    print(labels[:5, :20])


if __name__ == "__main__":
    main()
