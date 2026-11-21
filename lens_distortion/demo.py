"""Demo: lens distortion on a synthetic grid image.

Builds a checkerboard-style grid, applies barrel distortion
(k1 = 0.3), corrects it back, and prints the mean squared error
between the original and the corrected image, plus the point
round-trip error of distort -> undistort.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lens_distortion.lens_distortion import (
    correct_image,
    distort_image,
    distort_points,
    undistort_points,
)


def make_grid(height=120, width=160, cell=20):
    """Black image with a white grid every ``cell`` pixels."""
    grid = np.zeros((height, width), dtype=np.uint8)
    grid[::cell, :] = 255
    grid[:, ::cell] = 255
    return grid


def main():
    grid = make_grid()
    k1, k2 = 0.3, 0.0
    distorted = distort_image(grid, k1, k2)
    corrected = correct_image(distorted, k1, k2)
    mse_distort = float(np.mean((distorted.astype(np.float64) - grid) ** 2))
    mse_roundtrip = float(
        np.mean((corrected.astype(np.float64) - grid) ** 2)
    )
    # Point round-trip: distort then undistort a ring of points.
    rng = np.random.default_rng(0)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    points = np.column_stack(
        [79.5 + 60 * np.cos(angles), 59.5 + 45 * np.sin(angles)]
    )
    warped = distort_points(points, k1, k2, image_shape=grid.shape)
    back = undistort_points(warped, k1, k2, image_shape=grid.shape)
    point_err = np.linalg.norm(back - points, axis=1)
    # Barrel distortion pushes points outward: check mean radius grows.
    centre = np.array([79.5, 59.5])
    r_before = np.linalg.norm(points - centre, axis=1).mean()
    r_after = np.linalg.norm(warped - centre, axis=1).mean()
    print(f"grid {grid.shape}, k1={k1}, k2={k2}")
    print(f"mean radius of ring points: {r_before:.2f} -> {r_after:.2f} px "
          f"(barrel: outward)")
    print(f"MSE original vs distorted: {mse_distort:.1f}")
    print(f"MSE original vs distort->correct: {mse_roundtrip:.2f}")
    print(f"point round-trip error: mean {point_err.mean():.4f} px, "
          f"max {point_err.max():.4f} px")
    return corrected


if __name__ == "__main__":
    main()
