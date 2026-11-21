# SMOTE Oversampler

SMOTE (Synthetic Minority Over-sampling Technique) from scratch, numpy
only. It fixes class imbalance by creating new synthetic minority
samples instead of just duplicating existing ones: for each minority
sample it picks one of its `k_neighbors` nearest neighbours **of the
same class** at random and places a new point on the line segment
between them, `x_i + gap * (x_neighbour - x_i)` with `gap` drawn from
Uniform[0, 1]. Every synthetic point is therefore a convex combination
of two real minority points and stays inside the minority region.
Multi-class datasets are supported — every class below the majority
count is raised to it.

## How to run

```bash
python smote_oversampler/demo.py
python -m pytest tests/test_smote_oversampler.py -q
```

## Usage

```python
from smote_oversampler import SMOTE, fit_resample

# Balance all classes up to the majority count (reproducible)
X_res, y_res = SMOTE(k_neighbors=5, random_state=42).fit_resample(X, y)

# Or use the convenience wrapper
X_res, y_res = fit_resample(X, y, random_state=42)

# Other strategies: float ratio, or exact counts per class
X_res, y_res = SMOTE(sampling_strategy=0.5, random_state=0).fit_resample(X, y)
X_res, y_res = SMOTE(sampling_strategy={1: 50}, random_state=0).fit_resample(X, y)

smote = SMOTE(random_state=0).fit(X, y)
smote.class_counts_   # {0: 100, 1: 10}
smote.n_synthetic_    # 90
```

## Example output

The demo builds two Gaussian blobs — 100 majority vs 10 minority
samples in 2D (seed 42) — and runs SMOTE:

```
Before SMOTE: class 0 = 100, class 1 = 10 (total 110)
After SMOTE:  class 0 = 100, class 1 = 100 (total 200)
Synthetic samples created: 90
Distance from each synthetic point to its nearest real minority sample: min 0.002, max 0.668
First 3 synthetic points:
  [4.203, 4.577]
  [3.797, 4.348]
  [3.093, 3.673]
```

The small nearest-neighbour distances confirm the synthetic points sit
right inside the minority blob, on segments between real minority
samples.

## Notes and limits

- `k_neighbors` must be at least 1 and smaller than the number of
  samples in the smallest class being oversampled; a class with a
  single sample cannot be SMOTEd (needs at least 2).
- A dict `sampling_strategy` target must be at least the class's
  current count; classes not listed in the dict are left alone.
- If the classes are already balanced, `fit_resample` returns copies of
  `X` and `y` unchanged. No downsampling is ever done.
- `X` must be a finite 2D float array and `y` a 1D label array (ints or
  strings); empty input, NaN/inf, and mismatched `X`/`y` lengths raise
  `ValueError`, as does calling `resample()` before `fit()`.
```

## Tests

27 pytest tests in `tests/test_smote_oversampler.py`: balancing to the
majority count, exact float-ratio and dict targets, multi-class and
string labels, synthetic points lying exactly on segments between real
minority samples, reproducibility with the same seed, different seeds
giving different results, balanced-input passthrough, and `ValueError`
checks for empty/1D/NaN/mismatched input, `k_neighbors=0`, `k_neighbors`
too large, singleton classes, dict targets below the current count, and
invalid `sampling_strategy` values.
