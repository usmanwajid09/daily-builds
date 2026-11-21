"""Demo: RANSAC homography on synthetic correspondences.

Generates a known homography (rotation + translation + a small
perspective term), projects 60 random points through it, adds noise,
replaces 20 of them with gross outliers, then recovers the homography
with RANSAC and prints the inlier count and the error against the
true homography.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ransac_homography.ransac_homography import (
    apply_homography,
    ransac_homography,
    reprojection_errors,
)


def make_data(seed=0, n_points=60, n_outliers=20, noise=0.5):
    """True homography, clean points, noisy correspondences, outliers."""
    rng = np.random.default_rng(seed)
    angle = np.deg2rad(12.0)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    true_h = np.array(
        [
            [cos_a, -sin_a, 25.0],
            [sin_a, cos_a, -15.0],
            [0.0004, 0.0002, 1.0],
        ]
    )
    src = rng.uniform(0, 400, size=(n_points, 2))
    dst_clean = apply_homography(true_h, src)
    dst = dst_clean + rng.normal(0.0, noise, size=dst_clean.shape)
    outlier_idx = rng.choice(n_points, size=n_outliers, replace=False)
    dst[outlier_idx] = rng.uniform(-100, 500, size=(n_outliers, 2))
    return true_h, src, dst, outlier_idx


def main():
    true_h, src, dst, outlier_idx = make_data()
    est_h, inliers, iterations = ransac_homography(
        src, dst, threshold=3.0, max_iterations=1000, min_inliers=40,
        seed=42,
    )
    # Compare up to scale: normalise both so H[2, 2] == 1 (already true).
    fro_error = np.linalg.norm(est_h - true_h)
    inlier_errors = reprojection_errors(est_h, src[inliers], dst[inliers])
    true_inliers = np.ones(len(src), dtype=bool)
    true_inliers[outlier_idx] = False
    precision = (inliers & true_inliers).sum() / max(inliers.sum(), 1)
    recall = (inliers & true_inliers).sum() / true_inliers.sum()
    print(f"correspondences: {len(src)} "
          f"({len(outlier_idx)} outliers injected)")
    print(f"RANSAC: {int(inliers.sum())} inliers found in "
          f"{iterations} iterations")
    print(f"inlier precision: {precision:.3f}, recall: {recall:.3f}")
    print(f"Frobenius error vs true H: {fro_error:.4f}")
    print(f"mean reprojection error on inliers: {inlier_errors.mean():.3f} px "
          f"(max {inlier_errors.max():.3f} px)")
    return est_h, inliers


if __name__ == "__main__":
    main()
