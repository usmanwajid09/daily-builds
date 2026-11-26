# Flood-Fill Segmentation

Stack-based flood fill / region growing on a 2D grayscale image
(numpy only, no OpenCV). A pixel joins the region when its value is
within `tolerance` of the seed pixel's value, with 4- or
8-connectivity. `fill_holes` fills background holes fully enclosed
by foreground pixels in a binary mask.

## Run

```bash
python flood_fill/demo.py
python -m pytest tests/test_flood_fill.py -q
```

## Usage

```python
import numpy as np
from flood_fill.model import flood_fill, region_grow, fill_holes

image = np.zeros((60, 80), dtype=np.uint8)
image[16:45, 11:40] = 100

mask = flood_fill(image, seed=(30, 25), tolerance=5.0, connectivity=4)
mask = region_grow(image, seed=(30, 25), tolerance=5.0)  # same result

ring = np.zeros((9, 9), dtype=bool)
ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
solid = fill_holes(ring)  # 9x9 all True
```

`flood_fill` / `region_grow` raise `ValueError`/`TypeError` for a
non-2D or non-numeric image, an out-of-bounds or non-integer seed, a
negative/non-finite tolerance, or a connectivity other than 4 or 8.

## Demo output

On a 60×80 image (background 0, disc of value 100, rectangle of
value 180), flood-filling from `(30, 25)` with tolerance 5.0:

```
image: 60x80, disc value 100, rectangle value 180
seed (30, 25) (value 100), tolerance 5.0, connectivity 4
filled pixels: 613
disc true area: 613 pixels
match: True (difference: 0)
ring mask: 9x9, foreground before fill_holes: 32, after: 81 (hole of 49 pixels filled)
```

The fill recovers the disc exactly (613 pixels) and never leaks
into the 180-valued rectangle; `fill_holes` turns the 9×9 ring
(32 foreground pixels) into a solid 81-pixel block.
```

