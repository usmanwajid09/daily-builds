# Affine Warp

Affine image warping from scratch (numpy only).

Applies translation, rotation, scaling and shear to grayscale or RGB
images using inverse mapping with bilinear interpolation: for each
output pixel, the inverse affine matrix finds where it came from in
the input, and the four surrounding input pixels are blended. Pixels
that map outside the input are filled with `fill_value`.

## Run

```bash
python affine_warp/demo.py
python -m pytest tests/test_affine_warp.py -q
```

## Example

Demo on a 60×80 image with a bright 20×20 square in the middle:

```
total brightness before: 102000
total brightness after 30 deg rotation: 101958
total brightness after translate (5, -3): 102000
rotated centre pixel: 255.0
rotated corner pixel: 0.0
```

Rotation preserves total brightness (up to interpolation error) and
the centre pixel stays bright.

## API

- `warp_affine(image, matrix, output_shape=None, fill_value=0.0)`
- `build_affine_matrix(translation, rotation_deg, scale, shear)` → 3×3
- `rotation_matrix(angle_deg, center)` → 3×3 rotation about a point
- `warp_translate(image, dx, dy, ...)`, `warp_rotate(image, angle_deg, ...)`
  and `warp_scale(image, scale, ...)` convenience wrappers
