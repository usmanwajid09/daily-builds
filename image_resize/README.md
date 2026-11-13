# Image Resize, from scratch

Image resizing with numpy only — no OpenCV, no PIL. Two interpolation
methods, for grayscale `(H, W)` and RGB `(H, W, 3)` images, up- or
downscaling:

- **nearest** — nearest-neighbour: each output pixel copies the closest
  input pixel. Fast and exact for integer scale factors, but blocky.
- **bilinear** — bilinear interpolation: each output pixel is a weighted
  average of its four surrounding input pixels. Smoother, and exact for
  a linear gradient.

Coordinates use the "align corners" convention, so resizing to the same
size returns the input unchanged.

## Usage

```bash
pip install -r requirements.txt
python -m image_resize.demo
```

This builds a synthetic 40x60 RGB gradient image, resizes it up to
80x120 and down to 10x15 with both methods, and prints the shapes plus
a sample pixel check.

## Run tests

```bash
python -m pytest tests/test_image_resize.py -v
```
