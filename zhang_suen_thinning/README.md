# Zhang-Suen Thinning, from scratch

Thins a binary image to a 1-pixel-wide skeleton that preserves the
shape's topology — numpy only, no OpenCV.

## Why

A skeleton keeps what matters about a shape (its strokes, branches,
and endpoints) and throws away the bulk. It is the standard front end
for handwriting and fingerprint recognition, road-network extraction,
and measuring the length of thin structures, where the width of a
stroke is noise and its centreline is the signal.

## Usage

```bash
pip install -r requirements.txt
python zhang_suen_thinning/demo.py
```

This builds a 20x30 binary image (a thick plus sign), thins it, and
prints the pixel counts before and after plus an ASCII rendering of
both.

Use it on your own image from Python:

```python
import numpy as np
from zhang_suen_thinning.thinning import zhang_suen_thin

binary = np.load("mask.npy")   # 2D bool, 0/1, or 0/255 array
skeleton = zhang_suen_thin(binary)  # 2D bool array, 1 px wide
```

The skeleton is always a subset of the input foreground, and thinning
a skeleton again changes nothing. `skeleton_pixel_count` counts the
skeleton pixels and `is_skeleton` reports whether an image is already
1 pixel wide.

## Run tests

```bash
python -m pytest tests/test_zhang_suen_thinning.py -v
```
