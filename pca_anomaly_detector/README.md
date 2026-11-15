# PCA Anomaly Detector

Anomaly detection from scratch using PCA reconstruction error. PCA is
fit on normal data; a new point is scored by how badly the
low-dimensional model rebuilds it. Points the model cannot rebuild
are flagged as anomalies. The threshold defaults to
``mean + 3 * std`` of the training errors (a quantile rule or an
explicit cutoff can be used instead).

## Why

Reconstruction error is the simplest unsupervised anomaly score and
the same idea behind autoencoder-based detectors: normal data lies
near a low-dimensional subspace, anomalies do not.

## Usage

```bash
pip install -r requirements.txt
python pca_anomaly_detector/demo.py
```

This fits on 200 correlated 2D points, then tests 20 normal points
plus 5 far outliers.

Example output:

```
Training samples: 200, components kept: 1, threshold: 0.1194
Mean train reconstruction error: 0.0192
Test set: 20 normal + 5 outliers; flagged as anomalies: 5
Outliers flagged: 5 / 5
Normal points flagged: 0 / 20 (false positives)
Max normal-point error: 0.0687, min outlier error: 15.7878
```

## Run tests

```bash
python -m pytest tests/test_pca_anomaly_detector.py -q
```
