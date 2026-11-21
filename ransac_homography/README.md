# ransac_homography

RANSAC homography estimation from point correspondences (numpy only).
A homography is a 3x3 matrix mapping points in one image plane to
another (planar scenes, pure camera rotation, a change of viewpoint
of a plane).

## API

- `compute_homography(src, dst)` — direct linear transform (DLT) on
  Hartley-normalised points; needs >= 4 correspondences, raises
  `ValueError` on degenerate (e.g. collinear) configurations.
- `apply_homography(H, points)` — map (N, 2) points through H.
- `reprojection_errors(H, src, dst)` — per-correspondence Euclidean
  error in pixels.
- `ransac_homography(src, dst, threshold=3.0, max_iterations=1000, min_inliers=None, seed=None)` — returns `(H, inliers, iterations)`;
  H is refit on all inliers, `inliers` is a boolean mask, and
  `iterations` is how many 4-point samples were tried. Raises
  `RuntimeError` if no model with >= 4 inliers is found.

## Run

```bash
python -m pytest tests/test_ransac_homography.py -q
python ransac_homography/demo.py
```

## Example output

```
correspondences: 60 (20 outliers injected)
RANSAC: 40 inliers found in 69 iterations
inlier precision: 1.000, recall: 1.000
Frobenius error vs true H: 0.5867
mean reprojection error on inliers: 0.604 px (max 1.055 px)
```
