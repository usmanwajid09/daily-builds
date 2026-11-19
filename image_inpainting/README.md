# image_inpainting

Image inpainting by iterative diffusion (numpy only).

A masked-out hole is filled by repeatedly replacing each unknown
pixel with the average of its four direct neighbours — the discrete
Laplace equation solved by Jacobi iteration. The fill smoothly
interpolates the surrounding known values, so a hole in a gradient is
recovered as a smooth continuation of that gradient. Works on
grayscale (2-D) and RGB (3-D) images; unknown pixels start at the
known-pixel mean and the iteration stops when the largest per-pixel
change drops below `tolerance`.

## API

- `inpaint(image, mask, max_iterations=1000, tolerance=1e-3)` —
  returns the filled array, same shape and dtype as `image`. `mask`
  is a 2-D boolean array (`True` = fill this pixel). Raises
  `ValueError` for bad shapes/dtypes, an empty image, or a mask that
  covers every pixel.
- `inpaint_with_info(image, mask, ...)` — as above, but returns
  `(filled, iterations, converged)`.

## How to run

```bash
python image_inpainting/demo.py
python -m pytest tests/test_image_inpainting.py -q
```

## Demo output

```text
grayscale (24, 24), hole 36 pixels
  inpainted in 31 iterations (converged=True)
  hole MAE=0.001, max error=0.003
  known pixels unchanged: True
RGB (24, 24, 3): hole MAE=0.001, max error=0.003
```

The demo gradient is linear, which is the exact solution of the
Laplace equation, so diffusion recovers the hole almost exactly.
14 tests pass.
