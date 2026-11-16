# fast_corner_detector

FAST corner detection (Features from Accelerated Segment Test), implemented
with NumPy only.

## What it does

`detect_corners(image, ...)` returns the `(row, col)` coordinates of the
corners in a grayscale image, plus a score for each corner (stronger,
higher-contrast corners score higher). It also ships `is_corner` and
`corner_score` for testing a single pixel.

## How FAST works

For every pixel, look at the 16 pixels on a Bresenham circle of radius 3
around it. The pixel is a corner if at least `n_contiguous` of those circle
pixels in a row are all brighter than `centre + threshold`, or all darker
than `centre - threshold`.

Two optimisations from the original FAST paper are included:

- **Pre-test:** pixels 1, 5, 9 and 13 on the circle are checked first. A
  contiguous run of `n` circle pixels always contains at least `n // 4` of
  these four cardinal pixels, so unless that many of them are brighter (or
  darker) than the centre, the pixel cannot be a corner and the full
  16-pixel test is skipped. For the default `n = 12` this is the classic
  "at least 3 of the 4" pre-test; for `n = 9` it is "at least 2 of the 4",
  which is what lets a 90-degree corner — whose cardinal pixels split
  2 bright / 2 dark — reach the full test.
- **Non-maximum suppression (NMS):** corners usually fire as a small cluster
  of neighbouring pixels, so after detection only the strongest corner
  within a small radius is kept.

The corner score is the largest sum of absolute differences from the centre
over a qualifying contiguous run of brighter or darker circle pixels.

## FAST-9 vs FAST-12

The default is `n_contiguous=12` (FAST-12), the stricter variant. A perfect
90-degree step corner produces at most an 11-pixel arc on the 16-pixel
circle, so FAST-12 only fires on sharper (acute) corners. For ordinary
square or rectangular corners use `n_contiguous=9` (FAST-9), the classic
setting.

## Usage

```python
import numpy as np
from fast_corner_detector.fast import detect_corners

image = np.zeros((100, 160))
image[20:50, 20:50] = 255  # white square on black

corners, scores = detect_corners(image, threshold=20, n_contiguous=9)
print(corners)  # e.g. [[20 20] [20 49] [49 20] [49 49]]
print(scores)   # e.g. [2805. 2805. 2805. 2805.]
```

Run the demo (prints the corners found on a synthetic square + triangle,
and 0 corners on a uniform image):

```bash
python fast_corner_detector/demo.py
```

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `threshold` | `20` | A circle pixel counts only if it differs from the centre by more than this. Must be > 0. Higher values give fewer, stronger corners. |
| `n_contiguous` | `12` | Minimum length of a contiguous brighter/darker run on the circle (1–16). 9 for ordinary corners, 12 for sharp ones. |
| `nms` | `True` | Collapse each cluster of corner pixels into one corner. |
| `nms_radius` | `3` | With NMS on, no two returned corners are closer than this (Chebyshev distance). |

Pixels within 3 pixels of the image border are skipped, since the
radius-3 circle must fit inside the image. Invalid input (a non-2D or
empty array, `threshold <= 0`, `n_contiguous` outside 1–16) raises
`ValueError` with a clear message.

## Tests

```bash
python -m pytest tests/test_fast_corner_detector.py -q
```
