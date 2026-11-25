# Laplacian Edge Detector, from scratch

A tiny second-derivative edge detector — no OpenCV, just numpy. It
convolves a grayscale (or RGB) image with a 3x3 Laplacian kernel,
which responds strongly where the intensity changes quickly and is
zero in flat regions. Two edge readouts are provided: thresholding
the response magnitude, and zero crossings (where the response
changes sign between neighbouring pixels, which localises an edge to
a single pixel). Replicate padding keeps the output the same size as
the input.

## Why

The Laplacian is the simplest second-derivative operator: unlike
first-derivative (gradient) detectors it gives a signed response, so
an edge shows up as a positive lobe on one side and a negative lobe
on the other, with the true edge sitting exactly at the zero
crossing between them. The 4-neighbour kernel uses only the cross
neighbours (centre weight -4); the 8-neighbour kernel uses all eight
(centre weight -8) and is a little more sensitive to diagonal edges.

## Usage

```bash
pip install -r requirements.txt
python laplacian_edge_detector/demo.py
```

This builds a synthetic 40x60 image (a bright disc and a bright
rectangle on a dark background), runs the Laplacian on it, and prints
the response statistics, the zero-crossing count, the edge-pixel
count above a threshold, and an ASCII zero-crossing map.

Use it on your own image from Python:

```python
import numpy as np
from laplacian_edge_detector.laplacian_edge_detector import (
    laplacian, zero_crossings, edge_mask,
)

image = np.load("photo_gray.npy")  # 2D array
response = laplacian(image)                    # signed response
edges = zero_crossings(response, threshold=50.0)  # boolean mask
strong = edge_mask(image, threshold=150.0)    # |response| >= 150
```

## Run tests

```bash
python -m pytest tests/test_laplacian_edge_detector.py -v
```
