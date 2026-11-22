# kNN Anomaly Detector, from scratch

kNN distance-based anomaly detection, built with numpy only — no
scikit-learn. The anomaly score of a point is the mean distance to
its `k` nearest neighbours in the training set: points far from their
neighbours are unusual. For training points the point itself is
excluded (it is at distance 0 from itself), so the score uses the `k`
nearest *other* points. The decision threshold is the
`(1 - contamination)`-quantile of the training scores, so a new point
is flagged as an anomaly when its score exceeds the threshold.
`score_samples` returns the negated score (higher means more normal),
matching the scikit-learn convention, and `predict` returns +1 for
inliers and -1 for anomalies.

## Usage

```bash
pip install -r requirements.txt
python knn_anomaly_detector/demo.py
```

This fits on a 200-point Gaussian blob and prints the threshold, the
inlier rate on fresh normal points and the detection rate on far-away
outliers.

## Run tests

```bash
python -m pytest tests/test_knn_anomaly_detector.py -v
```

## Demo output

```
Train size: 200, threshold: 0.458
Fresh inliers classified as normal: 0.88
Outliers detected as anomalies: 1.00
Mean anomaly score (inliers): 0.278
Mean anomaly score (outliers): 8.982
```
