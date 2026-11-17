# document_scanner

Document scanning from scratch, numpy only — no OpenCV, no PIL.

Given a photo of a bright document (a sheet of paper) on a dark
background, it finds the document's four corners and perspective-warps
it to a flat rectangle, as if it had been scanned.

## How it works

1. **Threshold** — Otsu's method (refined to the midpoint of the two
   class means) separates the bright document from the dark background;
   isolated single-pixel noise is dropped.
2. **Corners** — the convex hull (monotone chain) of the bright region
   is taken, and its four extreme points are ordered TL, TR, BR, BL.
3. **Warp** — the 8×8 linear system for the homography from the flat
   rectangle to the detected corners is solved, then the image is
   inverse-warped with bilinear interpolation.

## API

```python
from document_scanner.document_scanner import (
    find_document_corners, warp_document, scan_document,
)

corners = find_document_corners(image)              # (4, 2) float array, TL/TR/BR/BL
warped = warp_document(image, corners, width=400, height=500)
warped, corners = scan_document(image, width=400, height=500)
```

`image` is a 2D grayscale or 3D RGB/RGBA numpy array. Raises `ValueError`
with a clear message for bad shapes, non-finite values, non-positive
output sizes, or when no bright quadrilateral is found (e.g. an
all-dark or all-bright image).

## How to run

From the repo root:

```bash
/home/hatch/workspace/db-venv/bin/python document_scanner/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_document_scanner.py -q
```

## Demo output

The demo builds a synthetic 200×200 photo (dark background ≈ 20, bright
rotated quadrilateral ≈ 220, fixed seed 42), detects its corners and
warps it to 120×160:

```text
True corners (TL, TR, BR, BL):
[[ 50.  40.]
 [160.  30.]
 [170. 150.]
 [ 40. 160.]]
Detected corners (TL, TR, BR, BL):
[[ 50.  40.]
 [160.  30.]
 [170. 150.]
 [ 40. 160.]]
Corner error per corner (px): [0. 0. 0. 0.]
Max corner error: 0.00 px
Warped shape: (160, 120)
Warped mean brightness: 217.2 / 255
Warped centre pixel: 219.4
Warped corner pixel (should be bright): 223.3
Exact-corner warp mean brightness: 217.2 / 255
```

## Tests

14 pytest tests in `tests/test_document_scanner.py` — corner recovery
accuracy, explicit threshold, warp shape/brightness, `scan_document`,
RGB input, homography correctness, corner ordering, and input
validation (all-dark / all-bright images, bad shapes, bad corners,
bad output sizes, bad threshold, identity warp). All 14 pass.
