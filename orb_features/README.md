# ORB Features

ORB-style feature detection and matching from scratch (numpy only):
FAST corners (16-pixel Bresenham circle, contiguous-arc test) plus
BRIEF-like binary descriptors (fixed random point-pair intensity
comparisons inside a patch), matched by brute-force Hamming distance
with Lowe's ratio test.

## Run

```bash
python orb_features/demo.py
python -m pytest tests/test_orb_features.py -q
```

## Example

Demo on a synthetic image shifted by (dr=5, dc=8):

```
Image 1: 7 keypoints, descriptors (7, 256)
Image 2: 7 keypoints, descriptors (7, 256)
Matches after ratio test: 7
Median recovered shift (dr, dc): [5.0, 8.0] (true shift: [5, 8])
Matches with exactly the true shift: 7/7
```

The known shift is recovered from the matched descriptors.

## API

- `extract_orb(image, threshold=20.0, n_contiguous=12, n_bits=256, patch_radius=15, max_keypoints=500, seed=0)` → (keypoints, descriptors)
- `detect_keypoints(image, ...)` → list of (row, col), strongest first
- `describe(image, keypoints, ...)` → (keypoints, bool descriptors)
- `make_descriptor_pairs(n_bits=256, patch_radius=15, seed=0)`
- `hamming_distance(desc1, desc2)` → pairwise distance matrix
- `match_descriptors(desc1, desc2, ratio=0.75)` → [(i, j, distance)], sorted
- `is_corner(image, row, col, ...)`, `corner_score(image, row, col, ...)`
