# Scharr Edge Detector, from scratch

A tiny edge-detection building block — no OpenCV, just numpy. It
convolves a grayscale image with the 3x3 Scharr kernels (weights
-3, -10, -3), which are tuned for better rotational symmetry than the
Sobel operator, and returns the gradient magnitude and direction.
Replicate padding keeps the output the same size as the input.

## Why

Scharr is the sharper cousin of Sobel: same 3x3 shape, but the
(-3, -10, -3) weights approximate the true derivative more accurately
for diagonal edges. Magnitude tells you *how strong* an edge is;
direction tells you *which way* it runs (0 = vertical edge, pi/2 =
horizontal edge).

## Usage

```bash
pip install -r requirements.txt
python scharr_edge_detector/demo.py
```

This builds a synthetic 40x60 image (a bright disc and a bright
rectangle on a dark background), runs Scharr edge detection, and
prints the gradient statistics, the edge-pixel count above a
threshold, and an ASCII edge map.

Use it on your own image from Python:

```python
import numpy as np
from scharr_edge_detector.scharr import scharr_edges, threshold_edges

image = np.load("photo_gray.npy")  # 2D array
magnitude, direction = scharr_edges(image)
edges = threshold_edges(magnitude, threshold=200.0)  # boolean mask
```

## Run tests

```bash
python -m pytest tests/test_scharr_edge_detector.py -v
```
