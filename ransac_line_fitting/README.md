# ransac_line_fitting

RANSAC line fitting on noisy 2-D points with outliers (numpy only).

Ordinary least squares is dragged off course by even a few gross
outliers, because every point pulls on the fit. RANSAC instead
samples two points at a time, scores each candidate line by how
many points lie within `threshold` of it, keeps the candidate with
the most inliers, and finally refits by least squares on the
inliers alone. With enough trials the true line is recovered even
when half the points are outliers. The line is stored as
`y = slope * x + intercept`; a sampled pair with identical x is
skipped, since a vertical line has no finite slope.

## API

- `ransac_line(points, n_trials=200, threshold=1.0, min_inliers=2,
  seed=None)` — returns a `RansacResult` with `slope`, `intercept`,
  boolean `inliers` mask, `n_inliers`, `n_trials`, and a
  `predict(x)` method. Raises `ValueError` when no candidate reaches
  `min_inliers` inliers.
- `least_squares_line(points)` — ordinary least-squares
  `(slope, intercept)` for comparison.

## How to run

```bash
python ransac_line_fitting/demo.py
python -m pytest tests/test_ransac_line_fitting.py -q
```

## Demo output

```text
80 points: 60 on y = 2x + 1 (noise 0.3) + 20 outliers
RANSAC:      slope=2.011, intercept=0.931, inliers=64/80
Least squares: slope=1.593, intercept=3.092
RANSAC recovered 60/60 true inliers with 4 false positives
```

Least squares is pulled from slope 2.0 to 1.59 by the outliers;
RANSAC recovers slope 2.01 and all 60 true inliers. 17 tests pass.
