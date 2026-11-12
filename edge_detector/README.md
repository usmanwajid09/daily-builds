# Sobel Edge Detection, from scratch

A tiny computer-vision building block — no OpenCV, just numpy. It finds
the outlines in a grayscale (or RGB) image with the classic Sobel
pipeline: gradients, magnitude, non-maximum suppression, threshold.

## Why

Edge detection is the first step of most CV systems — object outlines,
lane markings, document corners. Implementing Sobel by hand makes the
pieces visible: why a step change gives a strong gradient, why flat
areas give nothing, and why thinning (non-maximum suppression) matters
if you want one-pixel-wide edges instead of a blurry band.

## Usage

```bash
pip install -r requirements.txt
python -m edge_detector.demo
```

This builds a synthetic 60x80 image (a bright disc and rectangle on a
dark background), detects its edges, and prints the edge-pixel count
plus an ASCII rendering of the outlines.

Use it on your own image from Python:

```python
import numpy as np
from edge_detector.edges import detect_edges

image = np.load("photo_gray.npy")   # 2D array, or (H, W, 3) RGB
edges = detect_edges(image, threshold=0.15)  # boolean edge map
```

`threshold` is relative to the strongest edge in the image (0–1):
lower values keep fainter edges, higher values keep only the strongest
outlines. `nms=False` skips the thinning step.

## Run tests

```bash
python -m pytest tests/test_edge_detector.py -v
```
