# perspective_warp

Perspective (homography) warp from scratch (numpy only).

A homography is a 3x3 matrix mapping points between two planes. It is
estimated from four point correspondences with the Direct Linear
Transform (DLT) plus Hartley normalisation, solved by SVD. Warping
uses inverse mapping — every output pixel is mapped back into the
source image and sampled with bilinear interpolation — so the output
has no holes. Grayscale (2D) and colour (H, W, C) images are supported.

## API

- `compute_homography(src_points, dst_points)` — 3x3 homography from
  four `(x, y)` correspondences, with `H[2, 2] == 1`. Raises
  `ValueError` for non-(4, 2), non-finite, or degenerate (collinear)
  points.
- `apply_homography(homography, points)` — maps `(N, 2)` points
  through a homography.
- `warp_perspective(image, homography, output_shape, fill_value=0.0)`
  — warped float64 image; pixels mapping outside the source get
  `fill_value`.

## How to run

```bash
python perspective_warp/demo.py
python -m pytest tests/test_perspective_warp.py -q
```

## Demo output

```
homography:
[[ 0.7436 -0.1282  5.    ]
 [-0.      0.7436 -0.    ]
 [-0.     -0.0066  1.    ]]
corner mapping max error: 1.91e-14
warped shape: (40, 40), min=0.0, max=273.0
warped centre pixel (20, 20): 200.0
warped top-left corner (0, 0): 0.0 (outside source -> fill 0)
round-trip mean abs error (interior): 1.189
```

The demo warps a 40x40 gradient image (bright 200 square in the
middle) from its corners onto a trapezoid. The homography maps the
corners onto the target to 1.91e-14, and warping back gives a mean
absolute error of 1.189 over the interior (bilinear resampling loss).
12 tests pass.
