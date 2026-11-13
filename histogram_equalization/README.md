# Histogram Equalization, from scratch

A tiny contrast-enhancement building block — no OpenCV, just numpy.
It spreads a washed-out image's intensities across the full 0–255
range using the cumulative distribution function of its histogram,
plus a simpler percentile-based contrast stretch as a helper.

## Why

Photos taken in fog, at night, or with a bad exposure bunch their
pixels into a narrow band of gray, so details wash out. Equalization
remaps intensities so every gray level gets roughly equal use, which
makes hidden detail visible; the stretch does the same thing linearly,
keeping the relative spacing of intensities.

## Usage

```bash
pip install -r requirements.txt
python -m histogram_equalization.demo
```

This builds a synthetic 40x60 low-contrast image (values ~100–140),
equalizes it, and prints min/max/mean/std before and after, plus the
contrast-stretch result for comparison.

Use it on your own image from Python:

```python
import numpy as np
from histogram_equalization.equalize import equalize_histogram, contrast_stretch

image = np.load("photo_gray.npy")   # 2D array, values in [0, 255]
equalized = equalize_histogram(image)        # uint8, same shape
stretched = contrast_stretch(image, 2, 98)  # 2nd-98th percentile -> 0-255
```

A constant image has no contrast to spread, so both functions return
it unchanged.

## Run tests

```bash
python -m pytest tests/test_histogram_equalization.py -v
```
