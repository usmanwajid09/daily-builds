# lens_distortion

Radial lens distortion — apply and correct — from scratch (numpy
only). Uses the Brown model with two radial coefficients (k1, k2):

    r_distorted = r * (1 + k1 * r^2 + k2 * r^4)

where r is the distance from the distortion centre, normalised by the
image's larger side so k1/k2 are scale-free. k1 > 0 is barrel
distortion (magnification grows with radius), k1 < 0 is pincushion.

## API

- `distort_points(points, k1, k2=0.0, centre=None, image_shape=None)` —
  apply distortion to (N, 2) pixel coordinates.
- `undistort_points(points, k1, k2=0.0, centre=None, image_shape=None, iterations=20)` — iterative inverse.
- `distort_image(image, k1, k2=0.0, centre=None)` — distorted image
  (inverse mapping + bilinear sampling).
- `correct_image(image, k1, k2=0.0, centre=None)` — undistorted image.

Images may be grayscale or RGB, uint8 or float. `centre` defaults to
the image centre.

## Run

```bash
python -m pytest tests/test_lens_distortion.py -q
python lens_distortion/demo.py
```

## Example output

```
grid (120, 160), k1=0.3, k2=0.0
mean radius of ring points: 52.77 -> 54.54 px (barrel: outward)
MSE original vs distorted: 6018.5
MSE original vs distort->correct: 1904.86
point round-trip error: mean 0.0000 px, max 0.0000 px
```
