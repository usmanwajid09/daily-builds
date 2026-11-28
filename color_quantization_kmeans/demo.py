"""Demo: shrink a noisy 4-colour image down to a 4-colour palette.

Builds a 48x64 RGB image from four solid colour blocks, adds noise,
quantises it to 4 colours with k-means and reports the recovered
palette and the reconstruction error.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from color_quantization_kmeans.quantization import (
    kmeans_quantize,
    quantization_mse,
)


def make_test_image(height=48, width=64, seed=0):
    """Return a noisy image built from four solid colour blocks."""
    rng = np.random.default_rng(seed)
    blocks = np.array([[200.0, 30.0, 30.0],
                       [30.0, 180.0, 60.0],
                       [40.0, 80.0, 220.0],
                       [230.0, 200.0, 40.0]])
    image = np.zeros((height, width, 3))
    half_h, half_w = height // 2, width // 2
    image[:half_h, :half_w] = blocks[0]
    image[:half_h, half_w:] = blocks[1]
    image[half_h:, :half_w] = blocks[2]
    image[half_h:, half_w:] = blocks[3]
    return image + rng.normal(0.0, 8.0, size=image.shape), blocks


def main():
    image, true_colors = make_test_image()
    quantized, palette = kmeans_quantize(image, n_colors=4, seed=0)
    mse = quantization_mse(image, quantized)
    unique = len(np.unique(quantized.reshape(-1, 3), axis=0))
    print(f"image: {image.shape[1]}x{image.shape[0]} RGB, palette size: 4")
    print(f"unique colours after quantisation: {unique}")
    print(f"reconstruction MSE: {mse:.2f}")
    print("recovered palette:")
    for colour in palette[np.argsort(palette.sum(axis=1))]:
        print(f"  RGB({colour[0]:.0f}, {colour[1]:.0f}, {colour[2]:.0f})")
    return quantized, palette


if __name__ == "__main__":
    main()
