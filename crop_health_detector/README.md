# Crop Health Detector

Detect crop health from synthetic multispectral (red + near-infrared) patches
using the standard vegetation indices **NDVI** and **EVI**, classify each
pixel as **healthy**, **stressed** or **bare soil**, and build a field health
map. Numpy + stdlib only; all data is generated in code (no external files).

## How it works

- `make_field(height, width, seed)` — generates a synthetic field split into
  a 2×2 grid of planted zones (top-left healthy, top-right stressed,
  bottom-left bare soil, bottom-right healthy) with realistic reflectance:
  healthy vegetation has high NIR (0.70–0.90) and low red (0.05–0.15),
  stressed vegetation lower NIR (0.45–0.55) and higher red (0.20–0.30),
  bare soil has NIR ≈ red so its NDVI sits near zero. Returns the red and
  NIR bands plus a ground-truth zone label map (0 = healthy, 1 = stressed,
  2 = bare). Deterministic for a given seed.
- `ndvi(red, nir)` — `(NIR − red) / (NIR + red)`, with safe division
  (0 where the denominator is 0). Healthy vegetation scores ~0.7–0.9.
- `evi(nir, red, blue=None)` — the 3-band EVI when a blue band is given,
  otherwise the 2-band EVI2: `2.5·(NIR − red) / (NIR + 2.4·red + 1)`.
- `classify_patch(ndvi, evi)` — classifies one patch (scalar or array, by
  mean) as `'healthy'` / `'stressed'` / `'bare'` using NDVI + EVI
  thresholds (healthy: NDVI ≥ 0.55 and EVI ≥ 0.40; stressed: NDVI ≥ 0.20
  and EVI ≥ 0.12).
- `health_map(ndvi, evi=None)` — classifies every pixel and returns a label
  map plus a summary of per-class counts, percentages and total pixels.
- `zone_mean_ndvi(ndvi, labels)` and `classification_accuracy(predicted, truth)`
  — per-zone mean NDVI and the fraction of pixels matching ground truth.

All functions validate their inputs and raise `ValueError` with a clear
message for mismatched band shapes, empty arrays, non-finite values,
negative or > 1 reflectance, NDVI outside [-1, 1], bad label codes, and
bad (non-ordered) thresholds.

## Usage

```python
from crop_health_detector.crop_health import (
    make_field, ndvi, evi, health_map, classification_accuracy, zone_mean_ndvi,
)

red, nir, truth = make_field(64, 64, seed=42)
labels, summary = health_map(ndvi(red, nir), evi(nir, red))
print(summary["percentages"])                      # {'healthy': 50.0, ...}
print(classification_accuracy(labels, truth))     # 1.0
print(zone_mean_ndvi(ndvi(red, nir), truth))      # {'healthy': 0.779, ...}
```

## Demo

Run from the repo root:

```
python crop_health_detector/demo.py
```

Output:

```
Crop Health Detector — synthetic multispectral field
Field: 64x64 px (seed=42), bands: red + NIR
NDVI range: [-0.077, 0.895]  mean: 0.473
EVI2 range: [-0.053, 1.052]  mean: 0.504

Mean NDVI per ground-truth zone:
  healthy  : 0.779
  stressed : 0.336
  bare     : -0.001

Health map (share of field):
  healthy  :  2048 px  ( 50.0%)
  stressed :  1024 px  ( 25.0%)
  bare     :  1024 px  ( 25.0%)

Classification accuracy vs ground truth: 100.0%
Healthy area: 50.0%  |  Stressed area: 25.0%  |  Bare soil: 25.0%
OK: accuracy >= 90% and healthy > stressed > bare NDVI ordering holds.
```

## Tests

27 pytest tests in `tests/test_crop_health_detector.py`, covering field
generation (shapes, ranges, zone NDVI signatures, determinism, invalid
sizes), NDVI/EVI values and formulas, patch classification, the full
health-map pipeline (100% end-to-end accuracy on the seeded field), and
invalid input (mismatched shapes, empty arrays, negative / > 1
reflectance, out-of-range NDVI, bad thresholds, bad label codes).

```
python -m pytest tests/test_crop_health_detector.py -q
# 27 passed
```
