"""Demo: Bayer demosaicing on a synthetic image.

Builds a smooth 40x60 RGB image (a radial red gradient, a green
ramp, and a blue checker-free gradient), samples it down to an RGGB
Bayer mosaic, demosaics it back, and prints the PSNR and the worst
per-channel error.
"""

import os
import sys

import numpy as np

# Allow running as `python demosaicing/demo.py` from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demosaicing.demosaicing import demosaic, mosaic, psnr


def make_test_image(height=40, width=60):
    """Smooth RGB gradients — friendly to bilinear interpolation."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    red = 255.0 * np.exp(-((yy - 20.0) ** 2 + (xx - 30.0) ** 2) / 800.0)
    green = 255.0 * (xx / width)
    blue = 255.0 * (yy / height)
    return np.stack([red, green, blue], axis=2)


def main():
    rgb = make_test_image()
    bayer = mosaic(rgb, pattern="RGGB")
    recovered = demosaic(bayer, pattern="RGGB")
    score = psnr(rgb, recovered)
    max_error = np.abs(rgb - recovered).max(axis=(0, 1))
    print(f"image: {rgb.shape[0]}x{rgb.shape[1]} RGB, smooth gradients")
    print(f"bayer mosaic: {bayer.shape}, RGGB pattern")
    print(f"recovered: {recovered.shape}, dtype {recovered.dtype}")
    print(f"PSNR (mosaic -> demosaic): {score:.2f} dB")
    print(
        f"max per-channel error: R={max_error[0]:.2f} "
        f"G={max_error[1]:.2f} B={max_error[2]:.2f} (of 255)"
    )
    return recovered


if __name__ == "__main__":
    main()
