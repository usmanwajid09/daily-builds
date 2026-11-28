# Colour Quantisation (k-means)

Colour quantisation via k-means clustering from scratch (numpy only).

An RGB image is reduced to a small palette of representative colours:
pixels are clustered in RGB space with k-means++ initialisation, and
each pixel is replaced by its cluster centre. Useful for palette-based
art, GIF-style colour reduction, and as a first step for segmentation
by colour.

## Run

```bash
python color_quantization_kmeans/demo.py
python -m pytest tests/test_color_quantization_kmeans.py -q
```

## Example output

Demo on a noisy 48×64 image built from four solid colour blocks,
quantised to 4 colours:

```
image: 64x48 RGB, palette size: 4
unique colours after quantisation: 4
reconstruction MSE: 64.05
recovered palette:
  RGB(199, 30, 30)
  RGB(30, 180, 60)
  RGB(41, 80, 220)
  RGB(230, 200, 40)
```

The recovered palette matches the four true block colours almost
exactly, and the image now uses only 4 distinct colours instead of
thousands of noisy ones. The MSE of 64.05 is just the σ=8 sensor
noise itself (8² = 64) — the quantisation adds no error beyond it.

## API

- `kmeans_quantize(image, n_colors=8, max_iter=100, tol=1e-4, seed=0, n_init=5)` → `(quantized, palette)`; `quantized` has the input shape, `palette` has shape `(n_colors, 3)`; `n_init` k-means++ restarts are run and the lowest-error one is kept
- `quantization_mse(image, quantized)` → mean squared error between the two images
