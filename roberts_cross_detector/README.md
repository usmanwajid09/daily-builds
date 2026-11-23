# Roberts Cross Edge Detector, from scratch

A tiny edge-detection building block — no OpenCV, just numpy. It
applies the two 2x2 Roberts cross kernels, which difference diagonally
adjacent pixels, and returns the gradient magnitude and direction.
Because a 2x2 kernel has no centre pixel, the output is one row and
one column smaller than the input: element (i, j) describes the 2x2
block whose top-left corner is (i, j).

## Why

Roberts is the smallest and fastest gradient operator — just two
subtractions per pixel — and it responds sharply to fine detail and
thin lines. The trade-off is noise sensitivity: with no averaging at
all, a single noisy pixel looks like an edge. It is the classic
textbook baseline that Sobel, Prewitt and Scharr all improve on.

## Usage

```bash
pip install -r requirements.txt
python roberts_cross_detector/demo.py
```

This builds a synthetic 40x60 image (a bright disc and a bright
rectangle on a dark background), runs Roberts cross edge detection,
and prints the gradient statistics, the edge-pixel count above a
threshold, and an ASCII edge map.

Use it on your own image from Python:

```python
import numpy as np
from roberts_cross_detector.roberts import roberts_edges, threshold_edges

image = np.load("photo_gray.npy")  # 2D array
magnitude, direction = roberts_edges(image)  # shape (H-1, W-1)
edges = threshold_edges(magnitude, threshold=100.0)  # boolean mask
```

## Run tests

```bash
python -m pytest tests/test_roberts_cross_detector.py -v
```
