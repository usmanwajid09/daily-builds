# CLAHE — Contrast-Limited Adaptive Histogram Equalization

Applies CLAHE to a grayscale image, implemented from scratch with NumPy
(no OpenCV, no scikit-image). Unlike global histogram equalization, which
uses one mapping for the whole image, CLAHE enhances contrast *locally*,
so a dark corner and a bright sky can both gain detail at once.

## How it works

1. **Split into tiles** — the image is divided into a grid of tiles
   (default 8×8).
2. **Per-tile histogram** — each tile gets its own 256-bin histogram.
3. **Clip the histogram** — bins taller than `clip_limit × mean bin count`
   are cut down, and the clipped pixels are redistributed evenly across
   all bins. This caps how steep any tile's mapping can get, so noise in
   flat regions is not amplified into speckles.
4. **CDF mapping** — each clipped histogram's cumulative distribution
   function becomes that tile's lookup table (darkest value → 0,
   brightest → 255).
5. **Bilinear blending** — each pixel's output is a weighted mix of the
   lookup tables of the four neighbouring tile centres, so tile
   boundaries stay invisible. Edge pixels use the nearest tile centre.

Flat tiles are left untouched (identity mapping), so a constant image
comes back unchanged.

## Usage

```python
import numpy as np
from clahe.clahe import apply_clahe

image = ...  # 2D grayscale array, uint8 or float
enhanced = apply_clahe(image)                       # defaults: 8x8 tiles, clip 2.0
enhanced = apply_clahe(image, tile_grid=(4, 4), clip_limit=3.0)
```

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `tile_grid` | `(8, 8)` | Number of tiles as `(rows, cols)`. Smaller tiles = more local adaptation, more noise risk. |
| `clip_limit` | `2.0` | Histogram clip limit, as a multiple of the mean bin count. Higher = more contrast (and more noise amplification). |

Input must be a 2D, non-empty array with finite values; anything else
raises `ValueError`. Output is `uint8` with the same shape as the input.

## Demo

```bash
python clahe/demo.py
```

Builds a synthetic low-contrast image (gradient + dark/bright patches +
noise, values ~90–150), runs CLAHE, and prints min/max/mean/std before
and after plus the contrast gain inside each patch.

## Tests

```bash
python -m pytest tests/test_clahe.py -q
```
