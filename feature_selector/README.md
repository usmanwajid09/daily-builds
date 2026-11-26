# Feature selector, from scratch

Feature selection utilities — no scikit-learn, just numpy:

- `variance_threshold` — drop near-constant features (unsupervised).
- `mutual_info_scores` — mutual information of each feature with a
  discrete target, estimated by binning (classification).
- `correlation_scores` — absolute Pearson correlation of each feature
  with a continuous target (regression).
- `select_top_k` — indices of the `k` highest-scoring features.

## Why

Most datasets carry noise features that only add variance and training
time. Scoring each feature against the target and keeping the best few is
the cheapest way to shrink a model without losing signal.

## Usage

```bash
pip install -r requirements.txt
python feature_selector/demo.py
```

This builds a synthetic dataset with 2 informative features and 4 noise
features, then shows the top-2 features selected by mutual information
are the informative ones.

## Demo output (seed 42)

```
Samples: 400, features: 6
Mutual information scores: [0.3746, 0.1203, 0.0181, 0.0235, 0.0127, 0.0136]
Correlation scores:        [0.7206, 0.4579, 0.1178, 0.0086, 0.0507, 0.0512]
Top-2 features by MI: [0, 1] (informative features are 0 and 1)
Features kept by variance threshold: 6 of 6
```

## Run tests

```bash
python -m pytest tests/test_feature_selector.py -v
```
