# Harris Corner Detection, from scratch

A tiny corner-finding building block — no OpenCV, just numpy. It finds
the corners in a grayscale (or RGB) image with the classic Harris &
Stephens pipeline: Sobel gradients, structure tensor, corner response
`R = det(M) - k * trace(M)²`, non-maximum suppression, top-N corners.

## Why

Corners are where an image changes in two directions at once, which
makes them stable landmarks: image stitching, tracking, and camera
calibration all start by finding them. Implementing Harris by hand
shows why corners score high (both eigenvalues of the structure tensor
large), edges score low or negative (one direction only), and flat
areas score ~0.

## Usage

```bash
pip install -r requirements.txt
python -m harris_corner.demo
```

This builds a synthetic 60x80 image (a bright rectangle on black),
detects its corners, and prints the coordinates found — four, near the
rectangle's corners.

Use it on your own image from Python:

```python
import numpy as np
from harris_corner.corners import detect_corners

image = np.load("photo_gray.npy")   # 2D array, or (H, W, 3) RGB
corners = detect_corners(image, max_corners=20)  # [(row, col), ...]
```

`k` is the Harris sensitivity (0.04–0.06 is the usual range);
`threshold_rel` keeps only corners at least that fraction as strong
as the strongest one; `nms_radius` merges corners closer than that
many pixels, keeping the stronger one.

## Run tests

```bash
python -m pytest tests/test_harris_corner.py -v
```
