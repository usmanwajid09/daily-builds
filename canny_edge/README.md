# Canny Edge Detection

Canny edge detection, implemented from scratch with numpy only — no
OpenCV, no scipy. Builds on the Sobel gradients from `edge_detector/`
and the Gaussian blur from `gaussian_blur/`, adding the two steps that
make Canny edges thin and clean: non-maximum suppression and double
thresholding with hysteresis.

## The pipeline

1. **Gaussian smoothing** — a separable 1D Gaussian kernel
   (`sigma = 1.0` by default) suppresses noise before any gradients are
   taken.
2. **Sobel gradients** — horizontal and vertical Sobel kernels give
   `gx`, `gy`, and the gradient magnitude.
3. **Non-maximum suppression** — each pixel survives only if its
   magnitude is a local maximum along its gradient direction (quantised
   to 0°, 45°, 90° or 135°), so edges come out one pixel wide.
4. **Double threshold** — pixels above the high threshold (default 30%
   of the strongest gradient) are strong edges; pixels between the low
   (10%) and high thresholds are weak edges.
5. **Hysteresis** — a weak edge is kept only if it connects
   (8-connected) to a strong edge, directly or through other weak
   edges. Isolated weak responses — usually noise — are dropped.

All convolutions use **replicate padding** (border values repeated), so
a flat image stays flat at the border and produces zero edges
everywhere, including at the borders. Zero padding would invent a fake
edge there.

## Usage

```python
import numpy as np
from canny_edge import canny_edges

image = np.load("image.npy")  # 2D grayscale, or (H, W, 3) RGB
edges = canny_edges(image)  # boolean array, same (H, W) shape
edges = canny_edges(image, sigma=1.5, low_ratio=0.1, high_ratio=0.3)

# The pipeline steps are public too:
from canny_edge import (
    gaussian_kernel_1d,
    sobel_gradients,
    non_maximum_suppression,
    hysteresis,
)

kernel = gaussian_kernel_1d(1.0)          # 1D array, sums to 1
gx, gy, magnitude = sobel_gradients(image)
thinned = non_maximum_suppression(magnitude, gx, gy)
edges = hysteresis(thinned, low=0.1 * thinned.max(), high=0.3 * thinned.max())
```

`sigma` must be > 0 and the thresholds must satisfy
`0 <= low_ratio <= high_ratio <= 1`; anything else raises `ValueError`
with a message saying what was wrong. Non-numeric, wrong-shaped,
empty, or non-finite images are rejected the same way.

## Run the demo

```bash
PYTHONDONTWRITEBYTECODE=1 python canny_edge/demo.py
```

Builds a synthetic 60x80 image (a bright disc and a bright rectangle on
black), runs Canny on it, and prints the edge pixel count plus a small
ASCII rendering of the edge map.

## Run the tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_canny_edge.py -q
```
