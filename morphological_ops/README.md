# Morphological Operations, from scratch

A tiny binary-image cleanup building block — no OpenCV, just numpy.
It implements erosion, dilation, opening, and closing on binary
images with a 3x3 (or custom) structuring element, built from shifted
copies of the image combined with AND / OR.

## Why

Real binary images are messy: thresholded scans have salt-and-pepper
specks, scanned text has broken strokes, and segmented cells have
pinholes. Morphology is the standard cleanup pass — opening deletes
specks smaller than the structuring element, closing fills holes and
gaps — and every later measurement (area, count, shape) depends on
the image being clean first.

## Usage

```bash
pip install -r requirements.txt
python -m morphological_ops.demo
```

This builds a synthetic 30x40 binary image (a solid rectangle plus
scattered noise specks and one hole), and prints the foreground pixel
count before and after erosion, dilation, opening, and closing.

Use it on your own image from Python:

```python
import numpy as np
from morphological_ops.morphology import opening, closing, structuring_element

mask = np.load("mask.npy")  # 2D boolean (or 0/1) array
clean = opening(mask)                          # specks removed
filled = closing(mask)                         # holes filled
cross = structuring_element("cross", size=3)   # plus-shaped element
clean = opening(mask, cross)
```

The border is treated as foreground for erosion and background for
dilation, so objects touching the image edge are not eaten away or
grown outward by the border itself.

## Run tests

```bash
python -m pytest tests/test_morphological_ops.py -v
```
