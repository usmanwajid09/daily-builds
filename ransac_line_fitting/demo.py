"""Demo: RANSAC line fitting on noisy points with outliers.

Generates 60 points on the line y = 2x + 1 with Gaussian noise
(std 0.3), plus 20 gross outliers scattered in the same range
(seed 0). Fits the line with RANSAC and with plain least squares
and prints both, showing how the outliers drag least squares away
while RANSAC recovers the true line.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ransac_line_fitting.ransac_line_fitting import (
    least_squares_line,
    ransac_line,
)


def make_points(n_inliers=60, n_outliers=20, seed=0):
    """Return (points, true inlier mask) for y = 2x + 1 with outliers."""
    rng = np.random.default_rng(seed)
    x_in = rng.uniform(0, 10, size=n_inliers)
    y_in = 2.0 * x_in + 1.0 + rng.normal(0, 0.3, size=n_inliers)
    x_out = rng.uniform(0, 10, size=n_outliers)
    y_out = rng.uniform(0, 25, size=n_outliers)
    points = np.column_stack([
        np.concatenate([x_in, x_out]),
        np.concatenate([y_in, y_out]),
    ])
    mask = np.zeros(len(points), dtype=bool)
    mask[:n_inliers] = True
    return points, mask


def main():
    points, true_mask = make_points()
    result = ransac_line(points, n_trials=200, threshold=1.0, seed=0)
    ls_slope, ls_intercept = least_squares_line(points)
    print(f"{len(points)} points: 60 on y = 2x + 1 (noise 0.3) + 20 outliers")
    print(f"RANSAC:      slope={result.slope:.3f}, intercept={result.intercept:.3f}, "
          f"inliers={result.n_inliers}/{len(points)}")
    print(f"Least squares: slope={ls_slope:.3f}, intercept={ls_intercept:.3f}")
    recovered = int((result.inliers & true_mask).sum())
    false_pos = int((result.inliers & ~true_mask).sum())
    print(f"RANSAC recovered {recovered}/60 true inliers "
          f"with {false_pos} false positives")


if __name__ == "__main__":
    main()
