# Seam Carving — Content-Aware Image Resizing

Shrink an image's width without squashing its content. Seam carving finds
connected paths of low-energy pixels (vertical "seams") and removes them one
at a time, so smooth background disappears while edges and objects survive.

## How it works

1. **Energy map** — each pixel's energy is the gradient magnitude
   `|dx| + |dy|` of the grayscale image (Sobel-style finite differences).
   Flat regions have near-zero energy; edges and objects have high energy.
2. **Dynamic programming** — a cumulative-energy table is built top to bottom:
   each pixel's cost is its own energy plus the minimum of the three pixels
   above it (left, straight, right).
3. **Backtracking** — starting from the lowest-cost pixel in the bottom row,
   walk back up, always stepping to the cheapest of the three neighbours
   above. The result is a connected seam (adjacent rows differ by at most
   one column) with minimum total energy.
4. **Removal** — delete the seam's pixels, recompute the energy, and repeat
   until the target width is reached.

## Usage

```python
import numpy as np
from seam_carving.seam_carving import (
    energy_map, find_vertical_seam, remove_vertical_seams, shrink_width,
)

image = np.random.rand(100, 120)          # grayscale, or (H, W, 3) for RGB

energy = energy_map(image)                # (100, 120) gradient magnitude
seam = find_vertical_seam(energy)         # one column index per row
narrower = remove_vertical_seams(image, 20)   # (100, 100)
target = shrink_width(image, 80)          # (100, 80)
```

Grayscale (2D) and RGB (3D) images are supported; energy for RGB is computed
on the luminance conversion. Invalid inputs (1D/empty arrays, negative `n`,
`target_width >= current width` or `<= 0`) raise `ValueError`.

## Demo

```bash
python seam_carving/demo.py
```

Builds a synthetic image (smooth background + bright rectangle), removes
10 vertical seams, and prints the shape change, the mean energy of the
removed seams versus the mean image energy, and proof that the bright
object is preserved.

## Tests

```bash
python -m pytest tests/test_seam_carving.py -q
```
