# Adaptive Threshold

Adaptive (local) thresholding for unevenly lit images (numpy only).

A global threshold fails when illumination varies across an image.
Adaptive thresholding compares each pixel against the local mean (or
a Gaussian-weighted local mean) of its neighbourhood minus a
constant `C`, so dark and bright regions are binarised independently.
A comparison helper runs both methods side by side.

## Run

```bash
python adaptive_threshold/demo.py
python -m pytest tests/test_adaptive_threshold.py -q
```

## Example

```python
from adaptive_threshold import adaptive_threshold, compare_thresholds

mask = adaptive_threshold(image, block_size=11, c=2.0, method="mean")
result = compare_thresholds(image, block_size=15, c=10.0)
# result: {"adaptive": ..., "global": ...,
#          "adaptive_foreground": int, "global_foreground": int,
#          "difference": int}
```

Demo on a 60×80 document-like image: dark bars ("ink") on a smooth
illumination ramp (100 on the left, 220 on the right):

```
image: 60x80 grayscale, mean=146.7, bars = 540 px
adaptive: ink found = 100.00%, paper kept = 99.95%
global:   ink found = 100.00%, paper kept = 60.14%
```

Both methods find the bars, but the global threshold calls the whole
dark left side "ink" — it keeps only 60% of the paper — while the
adaptive threshold keeps 99.95% of the paper on both sides.

## API

- `adaptive_threshold(image, block_size=11, c=2.0, method="mean")` → boolean mask; `method` is `"mean"` or `"gaussian"`
- `global_threshold(image, method="mean")` → boolean mask; `method` is `"mean"` or `"median"`
- `compare_thresholds(image, block_size=11, c=2.0, method="mean")` → dict with both masks and foreground counts
