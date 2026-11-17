# Superpixel Segmentation (SLIC, from scratch)

SLIC-style superpixel segmentation implemented from scratch with NumPy
only — no OpenCV, no scikit-image. A superpixel is a small group of
neighbouring pixels with similar colour; they are a standard first step
for image segmentation, object detection and other vision pipelines
because they cut the number of primitives from millions of pixels to a
few hundred regions while keeping object boundaries.

## How it works

1. **Grid initialisation** — cluster centres are seeded on a regular grid
   (spacing `S = sqrt(N / K)`) and each is nudged to the lowest-gradient
   pixel in its 3×3 neighbourhood, so centres start away from edges.
2. **Localised k-means** — each pixel is compared only with the centres
   whose 2S × 2S window covers it, using
   `D = sqrt(dc² + (m/S)² · ds²)`, where `dc` is colour distance, `ds`
   spatial distance and `m` is the `compactness` parameter. Centres are
   then updated to the mean of their pixels, until they stop moving or
   `max_iter` is reached.
3. **Connectivity enforcement** — a flood-fill pass guarantees every
   label is exactly one connected region; tiny fragments are merged into
   the most common neighbouring label.

## API

```python
from superpixel_segmentation.superpixel_segmentation import slic

labels, centers = slic(image, n_segments=100, compactness=10.0,
                       max_iter=10, seed=0)
```

- `image`: 2D grayscale or 3D RGB array, float in `[0, 1]` or `uint8`.
- `n_segments`: target number of superpixels (≥ 2, ≤ number of pixels).
- `compactness`: > 0; larger = more compact, grid-like superpixels,
  smaller = superpixels that follow colour boundaries more closely.
- Returns `labels` — an `(H, W)` int array with ids `0..K-1` — and
  `centers` — a `(K, 2 + C)` array of `[y, x, channel values...]`.

`enforce_connectivity(labels, min_size=None)` is also exported for
repairing label maps produced elsewhere. Bad inputs raise `ValueError`
with a clear message (wrong dimensionality, wrong channel count, float
values outside `[0, 1]`, `n_segments` out of range, `compactness <= 0`,
`max_iter < 1`).

## Run

From the repo root:

```bash
python superpixel_segmentation/demo.py
python -m pytest tests/test_superpixel_segmentation.py -q
```

## Demo output

```
Image shape: (60, 80, 3)
Requested segments: 16, produced segments: 16
Labels shape: (60, 80), dtype: int64
Segment size: mean=300.0, min=290, max=315 pixels (ideal = 300)
Centers shape: (16, 5) (first center y=21.8, x=69.3, colour=[0.11 0.8  0.1 ])
Superpixel border pixels: 458, of which 139 lie on a true colour edge (30.3%)
Mean within-segment colour std: 0.0298 (lower = purer segments)
First 5 rows of labels:
[[2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
 ...
```

(The exact numbers above are from the run at build time; rerunning the
demo reproduces them, since the seed is fixed.)

The demo builds a 60×80 synthetic image with four differently coloured
quadrants plus light noise, segments it into 16 superpixels and prints
label statistics, boundary adherence (how much of the superpixel border
sits on a true colour edge) and colour purity (mean within-segment
colour standard deviation).
