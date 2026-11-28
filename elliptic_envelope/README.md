# Elliptic Envelope

Robust-covariance (MCD-style) anomaly detection built from scratch with numpy only.

## What it does

Fits an ellipse to the bulk of the data using the Minimum Covariance
Determinant idea: many random subsets are drawn, each is refined by
"concentration steps" (keep the `h` points closest to the current
centre, refit), and the subset with the smallest covariance
determinant wins. The raw MCD covariance is rescaled with the
standard consistency factor, then a reweighting step refits on the
points inside the chi-square cut-off. Because outliers cannot drag
the centre or the covariance, the Mahalanobis distance to the robust
centre is a reliable outlier score. Points whose squared distance
exceeds the chi-square threshold for `1 - contamination` are flagged
-1; the rest are 1. The chi-square quantile and CDF are computed with
the Wilson-Hilferty approximation and a regularised incomplete gamma
function — no scipy needed.

## How to run

```bash
python elliptic_envelope/demo.py
```

```python
from elliptic_envelope.model import EllipticEnvelope

model = EllipticEnvelope(contamination=0.05, random_state=42)
model.fit(X)
print(model.predict(X_new))          # 1 = inlier, -1 = outlier
print(model.decision_function(X_new))
print(model.score_samples(X_new))    # higher = more normal
```

## Demo result

200 clean 2D Gaussian points plus 10 far outliers (centred at (8,8)),
`contamination=0.05`.

- Outliers detected: **10/10 (recall 1.00)**
- Inliers kept: **192/200 (accuracy 0.96)**
- Total flagged: 18

## Example output

```
Outliers injected:   10
Outliers detected:   10 (recall 1.00)
Inliers kept:        192/200 (accuracy 0.96)
Total flagged:       18
Threshold (chi2):    5.94
Robust centre:       [-0.05, -0.051]
First 5 predictions: [1, 1, 1, 1, 1]
First 5 true labels:  [1, 1, 1, 1, 1]
```

## Tests

11 pytest tests in `tests/test_elliptic_envelope.py` — outlier
recall, inlier retention, predictions in {1, -1}, outliers scoring
lower than inliers, decision-function sign matching predict, robust
centre near the true centre, Mahalanobis distance growing with
distance, and input validation.
