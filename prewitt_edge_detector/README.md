# Prewitt Edge Detector, from scratch

A tiny edge-detection building block — no OpenCV, just numpy. It
convolves a grayscale image with the 3x3 Prewitt kernels (uniform
-1, -1, -1 weights), which average the neighbouring rows/columns while
differentiating, and returns the gradient magnitude and direction.
Includes thresholding to turn the magnitude into a binary edge mask.
Replicate padding keeps the output the same size as the input.

## Why

Prewitt is the simplest 3x3 gradient operator: it is Sobel with the
centre weight dropped from 2 to 1, so it smooths slightly more and is
a touch less sensitive to noise. Magnitude tells you *how strong* an
edge is; direction tells you *which way* it runs.

## Usage

```bash
pip install -r requirements.txt
python prewitt_edge_detector/demo.py
```

This builds a synthetic 40x60 image (a bright disc and a bright
rectangle on a dark background), runs Prewitt edge detection, and
prints the gradient statistics, the edge-pixel count above a
threshold, and an ASCII edge map.

Use it on your own image from Python:

```python
import numpy as np
from prewitt_edge_detector.prewitt import prewitt_edges, threshold_edges

image = np.load("photo_gray.npy")  # 2D array
magnitude, direction = prewitt_edges(image)
edges = threshold_edges(magnitude, threshold=150.0)  # boolean mask
```

## Run tests

```bash
python -m pytest tests/test_prewitt_edge_detector.py -v
```
