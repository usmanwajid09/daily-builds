# Colour-Moment Classifier

Image classifier from scratch: per-channel colour moments
(mean / standard deviation / skewness per RGB channel) give a
9-dim feature vector, classified with k-nearest neighbours.
Numpy only — no scikit-learn, no data files.

Synthetic colour scenes are generated in code: **sunset** (warm
red/orange, with a bright horizontal band), **forest** (green
dominant) and **ocean** (blue dominant), each with light per-pixel
noise so no two seeds are identical.

## What it does

- `color_moments(image)` — 9-vector: for each RGB channel, mean,
  std, skewness (skewness = E[(x − mean)³] / std³, 0.0 when the
  channel is constant). Accepts float images in [0, 1] or uint8
  in [0, 255]; raises `ValueError` on bad shape, wrong channel
  count, empty, non-finite or out-of-range input.
- `extract_features(image)` — wrapper around `color_moments`.
- `generate_scene(kind, size=32, seed=None)` — synthetic RGB float
  image in [0, 1]; `kind` is one of `"sunset"`, `"forest"`, `"ocean"`.
- `ColorMomentClassifier(k=3)` — `fit(X, y)` on (n, 9) feature
  vectors (features are standardised per feature, std == 0 guarded
  to 1.0), `predict(X)` by majority vote among the k nearest
  neighbours in standardised space (ties broken by smallest label
  order), `score(X, y)` accuracy. `predict` before `fit`, invalid
  k (0 or > n_train) and mismatched X/y lengths all raise
  `ValueError`.
- `train_classifier(images, labels, k=3)` / `predict(images, classifier)`
  — convenience helpers that take raw images.

## How to run

```bash
python color_moment_classifier/demo.py
```

The demo trains on 15 synthetic scenes per kind (45 total, seeds
0–14) and tests on 9 fresh scenes per kind (27 total, seeds
100–108), with k=3.

## Demo result

```
Training set: 45 images (9-dim colour-moment features)
Test set: 27 images
Classifier: k=3, classes=['forest', 'ocean', 'sunset']
Overall test accuracy: 100.00% (27/27 correct)
  sunset: 100.00% (9/9 correct)
  forest: 100.00% (9/9 correct)
  ocean : 100.00% (9/9 correct)
```

Feature vector of one sunset image (seed 0), rounded to 3 decimals:

```
[0.911, 0.043, 0.593, 0.477, 0.135, 1.767, 0.155, 0.088, 1.565]
```

(R/G/B means 0.911 / 0.477 / 0.155 — red-dominant, as expected for
a sunset; the bright band shows up as positive skew in G and B.)

## Run tests

```bash
python -m pytest tests/test_color_moment_classifier.py -q
```

13 tests, all passing.
