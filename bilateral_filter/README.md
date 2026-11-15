# Bilateral Filter

Bilateral (edge-preserving) smoothing from scratch (numpy only).

Each neighbour's weight is the product of a spatial Gaussian (how
close the pixel is) and a range Gaussian (how similar its intensity
is), so flat regions are smoothed while sharp edges are kept. Works
on grayscale images and RGB images.

## Run

```bash
python bilateral_filter/demo.py
python -m pytest tests/test_bilateral_filter.py -q
```

## Example

Demo on a noisy 40×60 step image (50 on the left, 200 on the right,
σ=10 noise):

```
flat-region std before: left=10.03 right=10.09
flat-region std after:  left=2.67 right=2.72
step height before: 148.9
step height after:  150.1
```

Noise in the flat regions is cut to about a quarter while the ~150-unit
step is preserved almost exactly — a plain blur would have smeared it.

## API

- `bilateral_filter(image, diameter=5, sigma_color=25.0, sigma_space=5.0)`
  → float64 array, same shape as the input
- `gaussian_kernel_2d(diameter, sigma)` → normalised 2D Gaussian kernel
