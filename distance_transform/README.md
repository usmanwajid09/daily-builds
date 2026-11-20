# Distance Transform

Two-pass chamfer distance transform (numpy only).

For every foreground pixel, the distance to the nearest background
pixel. Exact Euclidean distance is expensive, so the transform is
approximated with two raster passes (forward, then backward) using
small integer masks: 3-4 (orthogonal 3, diagonal 4) or 5-7-11
(orthogonal 5, diagonal 7, knight move 11 — the more accurate one).
The nearest background pixel of every pixel is tracked as well.

## Run

```bash
python distance_transform/demo.py
python -m pytest tests/test_distance_transform.py -q
```

## Example

```python
from distance_transform import distance_transform, nearest_background

dist = distance_transform(binary, mask="chamfer5711")  # (H, W) float64
rows, cols = nearest_background(binary)  # nearest background per pixel
```

Demo on a 40×50 image with a filled disc (radius 12) as foreground:

```
chamfer 3-4:    max distance = 11.67 (disc centre), centre value = 11.67
chamfer 5-7-11: max distance = 12.00, centre value = 12.00 (true radius = 12)
nearest background of disc centre (20, 25): (9, 20)
background pixels have distance 0: True
```

The 5-7-11 mask recovers the disc radius exactly; the 3-4 mask
underestimates it slightly (11.67 vs 12) — the usual price of the
cheaper mask. The centre's nearest background (9, 20) is one of
several background pixels tied at chamfer distance 12.

## API

- `distance_transform(binary, mask="chamfer34", scale=True)` → `(H, W)` float64 distances in approximate pixel units (`scale=False` returns raw chamfer units)
- `nearest_background(binary, mask="chamfer34")` → `(nearest_row, nearest_col)` int64 arrays; background pixels map to themselves
