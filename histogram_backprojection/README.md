# Histogram Backprojection

Colour-based object localisation by histogram backprojection,
implemented from scratch with numpy only — no OpenCV.

A 3D RGB histogram is built from a region of interest, then every
pixel of a (new) image is scored by the histogram value of its colour.
The resulting probability map peaks where the image matches the model's
colour, so its peak or centroid localises the object — the same idea
that powers CamShift trackers and colour-based segmentation.

## Usage

```python
import numpy as np
from histogram_backprojection import build_histogram, localise

image = np.load("image.npy")  # (H, W, 3), values in [0, 255]
hist = build_histogram(image, roi=(x, y, w, h), bins=16)
result = localise(image, hist, threshold=0.5)
result["peak"]      # (x, y) of the highest-probability pixel
result["centroid"]  # weighted centroid of pixels above threshold * max
result["peak_value"]  # 1.0 (the map is scaled so its max is 1)
```

`backproject(image, hist)` returns the probability map in [0, 1]
(scaled so its maximum is 1). `locate_peak(prob_map)` and
`locate_centroid(prob_map, threshold=0.5)` work on any 2D probability
map. Images must be `(H, W, 3)` with values in [0, 255]; the ROI must
lie inside the image; `bins` must be in [2, 64]; an all-zero histogram
raises `ValueError`.

## Run the demo

```bash
PYTHONDONTWRITEBYTECODE=1 python histogram_backprojection/demo.py
```

Builds an image with a red disc and a blue distractor on green,
backprojects a histogram taken from the red disc, and prints the peak
and centroid next to the true disc centre.

## Run the tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_histogram_backprojection.py -q
```
