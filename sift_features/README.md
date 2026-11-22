# sift_features

A simplified SIFT-like feature detector and descriptor, implemented from
scratch with **NumPy only** — no OpenCV, no other libraries. It follows the
core ideas of Lowe's SIFT:

1. **Gaussian scale-space / DoG** — a pyramid of separable 1-D Gaussian
   blurs (edge padding) over 3 octaves; Difference of Gaussians between
   adjacent blur levels.
2. **Keypoint detection** — local extrema of the DoG over a 3x3x3
   neighbourhood (space x scale), kept only above a contrast threshold and
   passing Lowe's edge-response test on the 2x2 Hessian
   (`trace²/det < (r+1)²/r`), plus a small non-maximum suppression step.
3. **Orientation assignment** — a 36-bin gradient-orientation histogram
   around each keypoint; the dominant bin becomes the keypoint orientation.
4. **Descriptor** — a 4x4 spatial grid x 8 orientation bins = 128-dim
   vector, sampled from a 16x16 patch rotated to the keypoint orientation,
   Gaussian-weighted, L2-normalised, clipped at 0.2 and re-normalised
   (the standard SIFT recipe).
5. **Matching** — brute-force nearest neighbours with Lowe's ratio test
   (`match_descriptors`, default ratio 0.75).

## API

```python
from sift_features.sift_features import SIFTFeatures, match_descriptors

sift = SIFTFeatures(n_octaves=3, n_scales=3, contrast_threshold=0.04,
                    edge_threshold=10.0, n_bins=8, patch_size=16)

keypoints = sift.detect(image)                       # list of dicts:
                                                     # y, x, scale, octave,
                                                     # orientation, response
descriptors = sift.compute_descriptors(image, keypoints)  # (N, 128)
keypoints, descriptors = sift.detect_and_describe(image)  # valid pairs only
matches = match_descriptors(desc1, desc2, ratio=0.75)     # [(i, j, distance)]
```

`image` must be a 2-D grayscale NumPy array, at least 32x32, with finite
values; anything else raises `ValueError` with a clear message. Invalid
parameters (non-positive sigma/octaves, ratio outside (0, 1], descriptor
arrays not shaped (N, 128)) also raise `ValueError`.

## How to run

From the repo root:

```bash
python sift_features/demo.py
python -m pytest tests/test_sift_features.py -q
```

## Demo output

The demo builds a 128x128 synthetic image (a bright square, a bright disc
and a smooth Gaussian blob on a black background), detects keypoints, then
shifts the image by (5, 7) pixels and matches descriptors between the two
versions:

```
Image shape: (128, 128)
Keypoints detected: 5
Descriptor shape: (5, 128)
Descriptor norms: min=1.0000 max=1.0000 mean=1.0000
First 3 keypoints (y, x, scale, orientation):
  y=22.0 x=22.0 scale=2.02 orientation=0.087
  y=22.0 x=42.0 scale=2.02 orientation=1.658
  y=42.0 x=22.0 scale=2.02 orientation=-1.484
Keypoints in shifted image: 5
Good matches (ratio test 0.75): 4
Match distances: min=0.0000 max=0.0000 mean=0.0000
First 3 matches (i, j, distance) with keypoint positions:
  (22.0, 22.0) -> (27.0, 29.0)  d=0.0000
  (22.0, 42.0) -> (27.0, 49.0)  d=0.0000
  (42.0, 22.0) -> (47.0, 29.0)  d=0.0000
```

The matched keypoints land exactly 5 rows / 7 columns from their originals,
as expected for the shift, and the descriptors match with distance 0. The
fifth keypoint (a symmetric square corner) is rejected by the ratio test
because another corner looks almost identical — that is the ratio test
working as intended.
```

## Tests

18 pytest tests in `tests/test_sift_features.py` covering: blank image (0
keypoints), bright blob (>= 1 keypoint), descriptor shape (N, 128) and unit
norm, finite/bounded descriptor values, identical images giving identical
descriptors, determinism, shifted-image matching (count and position),
ratio-test rejection of random descriptors, invalid inputs (1-D, empty,
too-small, NaN images, bad ratio, bad `n_octaves`), and the Gaussian kernel
(normalised, symmetric). All 18 pass.
