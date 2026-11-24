# Image Moments

Raw, central, normalised and Hu moments for binary/grayscale images,
implemented from scratch with numpy only — no OpenCV.

Moments summarise a shape: `m00` is the area (pixel count for a binary
image), the first-order moments give the centroid, the second-order
central moments give the orientation, and the seven Hu invariants are
(approximately) unchanged by translation, scale and rotation — so they
work as a compact shape descriptor for matching and classification.

## Usage

```python
import numpy as np
from image_moments import raw_moments, centroid, orientation, hu_moments

image = np.zeros((60, 80))
image[20:40, 30:50] = 1.0  # a filled rectangle

raw_moments(image)["m00"]   # area = 400.0
centroid(image)             # (39.5, 29.5)
orientation(image)          # ~0.0 rad (wider than tall)
hu_moments(image)           # length-7 array of Hu invariants
```

`raw_moments(image, order=3)`, `central_moments` and
`normalised_moments` return dicts keyed `"m00"`, `"mu20"`, `"eta11"`,
etc. `moments_summary(image)` returns area, centroid, orientation,
Hu moments and the raw moments in one dict.

Images must be 2D, finite and non-negative; an all-zero image has no
centroid, so `centroid` / `central_moments` / `hu_moments` raise
`ValueError` for it. `order` must be an integer in [0, 3].

## Run the demo

```bash
PYTHONDONTWRITEBYTECODE=1 python image_moments/demo.py
```

Builds a synthetic binary image (disc + rectangle), prints area,
centroid, orientation and the seven Hu moments, then re-computes them
after translating the shape and prints the largest relative change.

## Run the tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_image_moments.py -q
```
