# Gamma Correction, from scratch

Pointwise intensity transforms for grayscale (2D) or RGB (H, W, 3)
images — no OpenCV, just numpy. Every function takes an image as a
numpy array (uint8, or float in [0, 1] or [0, 255]) and returns a
float64 array of the same shape with values in [0, 255].

## Why gamma matters

Human vision is not linear: we notice small changes in dark tones
far more than equal-sized changes in bright tones, so cameras and
displays store brightness on a power-law (gamma) curve instead of
linearly. A gamma of ~2.2 is the standard for encoding (sRGB), and
applying the inverse (~1/2.2) decodes it back for display. Getting
gamma wrong makes an image look washed out (too bright) or crushed
(too dark) even when every pixel value is "correct" — which is why
gamma correction is the first fix when an image's brightness just
looks off. The log transform does a similar job for shadow detail,
and a percentile contrast stretch rescues low-contrast images by
spreading the tones they actually use across the full range.

## Functions

- `gamma_correct(image, gamma)` — `255 * (image / 255) ** gamma`.
  `gamma < 1` brightens (lifts mid-tones), `gamma > 1` darkens,
  `gamma == 1` returns the image unchanged.
- `log_transform(image, c=1.0)` — normalised log transform:
  `255 * log(1 + image) / log(1 + 255)` for `c == 1`, and
  `255 * log(1 + c * image / 255) / log(1 + c)` in general. Monotonic,
  maps 0 to 0 and 255 to 255; expands dark tones, compresses bright
  ones.
- `contrast_stretch(image, low=2, high=98)` — maps the `low`/`high`
  percentiles linearly to [0, 255], clipping outside. Percentiles are
  computed per channel for RGB (globally for grayscale). A flat image
  (both percentiles equal) is returned unchanged.
- `adjust_brightness_contrast(image, brightness=0.0, contrast=1.0)` —
  `(image - 127.5) * contrast + 127.5 + brightness`, clipped to
  [0, 255].

All four validate their inputs and raise `ValueError` with a clear
message on a non-numeric, empty, 1D, or wrong-channel image, on
non-finite values, and on bad parameters (non-positive gamma / `c` /
contrast, non-finite brightness, or a percentile range that does not
satisfy `0 <= low < high <= 100`).

## Usage

```bash
pip install -r requirements.txt
python gamma_correction/demo.py
```

This builds a synthetic 64x64 gradient (dark left, bright right)
plus a small RGB test image, applies gamma 0.5 and 2.2, a log
transform, a contrast stretch, and a brightness/contrast adjustment,
and prints the means so you can see gamma 0.5 brighten and gamma
2.2 darken.

Use it on your own image from Python:

```python
import numpy as np
from gamma_correction import gamma_correct, log_transform, contrast_stretch, adjust_brightness_contrast

image = np.load("photo_gray.npy")   # 2D array, or (H, W, 3) RGB
bright = gamma_correct(image, 0.5)        # lift the shadows
stretched = contrast_stretch(image)       # spread tones over [0, 255]
adjusted = adjust_brightness_contrast(image, brightness=20.0, contrast=1.2)
```

## Run tests

```bash
python -m pytest tests/test_gamma_correction.py -v
```
