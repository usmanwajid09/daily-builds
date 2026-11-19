# High-Boost Filter, from scratch

Sharpens an image by adding a scaled copy of its high-frequency detail
back onto itself — `original + k * (original - blurred)` — numpy only,
no OpenCV.

## Why

Blurring removes edges and fine detail; the difference between the
original and its blur is exactly that detail (the unsharp mask). Adding
it back with a gain `k` makes edges crisper without inventing new
content, which is why unsharp masking is the standard first sharpening
step in photo editors and pre-processing pipelines. `k = 1` is classic
unsharp masking; larger `k` boosts harder.

## Usage

```bash
pip install -r requirements.txt
python high_boost_filter/demo.py
```

This builds a synthetic 40x60 image (a soft vertical step from 50 to
200), sharpens it with k=2, and prints pixel statistics plus the edge
profile before and after.

Use it on your own image from Python:

```python
import numpy as np
from high_boost_filter.filter import high_boost_filter

image = np.load("photo_gray.npy")   # 2D array, or (H, W, 3) RGB, 0-255
sharp = high_boost_filter(image, k=2.0, sigma=1.0)  # same shape, float64
```

Output is clipped to 0-255. `sharpening_mask(image, sigma)` returns
just the `original - blurred` detail layer.

## Run tests

```bash
python -m pytest tests/test_high_boost_filter.py -v
```
