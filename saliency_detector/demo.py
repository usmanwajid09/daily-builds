"""Demo: find the odd one out in a textured scene.

Builds a 64x64 image of random texture with one bright disk planted
in it, runs :func:`saliency_detector.saliency.spectral_residual_saliency`
and reports where the saliency peak lands versus where the disk
actually is.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from saliency_detector.saliency import saliency_peak, spectral_residual_saliency


def make_test_image(height=64, width=64, seed=0, centre=(32, 40), radius=8):
    """Return (smooth textured image with a bright disk, disk centre)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    image = (100.0 + 20.0 * np.sin(2 * np.pi * xx / 32.0)
             + 15.0 * np.cos(2 * np.pi * yy / 24.0)
             + rng.normal(0.0, 5.0, size=(height, width)))
    disk = (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius ** 2
    image[disk] = 230.0
    return image, centre


def main():
    image, centre = make_test_image()
    saliency = spectral_residual_saliency(image)
    peak = saliency_peak(saliency)
    distance = float(np.hypot(peak[0] - centre[0], peak[1] - centre[1]))
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, disk at {centre} (radius 8)")
    print(f"saliency range: min={saliency.min():.3f} max={saliency.max():.3f}")
    print(f"saliency peak at: {peak}")
    print(f"distance from disk centre: {distance:.1f} px")
    return saliency


if __name__ == "__main__":
    main()
