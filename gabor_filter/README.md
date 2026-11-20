# Gabor Filter Bank

Orientation- and scale-selective texture analysis from scratch
(numpy only).

A 2D Gabor filter is a sinusoidal plane wave multiplied by a Gaussian
envelope. A bank of filters at several orientations and wavelengths
responds strongly to edges and textures aligned with each filter, so
the response maps (and their local energy) are classic texture
features — e.g. for texture classification or segmentation.

## Run

```bash
python gabor_filter/demo.py
python -m pytest tests/test_gabor_filter.py -q
```

## Example

```python
from gabor_filter import gabor_bank, apply_gabor_bank, texture_energy

kernels = gabor_bank(n_orientations=4, wavelengths=(8.0, 16.0))
responses = apply_gabor_bank(image, kernels)   # (H, W, 8) response maps
energy = texture_energy(responses, window=9)   # local RMS energy per filter
```

Demo on a 64×64 image (vertical stripes left, horizontal stripes
right):

```
bank: 8 filters (4 orientations x 2 scales), response shape (64, 64, 8)
  wavelength=   8 theta=    0deg mean |response| =   23.19
  wavelength=   8 theta=   45deg mean |response| =    1.25
  wavelength=   8 theta=   90deg mean |response| =   23.72
  wavelength=   8 theta=  135deg mean |response| =    1.25
theta=0deg energy: left (vertical stripes) = 50.38, right (horizontal stripes) = 4.92
```

The 0° filter fires on the vertical stripes and is nearly silent on
the horizontal ones — exactly the orientation selectivity a Gabor bank
is for. Kernels are L1-normalised, so response magnitudes are on the
scale of the image values.

## API

- `gabor_kernel(ksize, sigma, theta, wavelength, gamma=0.5, psi=0.0)` → zero-mean `(ksize, ksize)` kernel
- `gabor_bank(n_orientations=8, wavelengths=(4.0, 8.0), ksize=21, sigma=4.0, gamma=0.5)` → list of kernels, scale-major
- `apply_gabor_bank(image, kernels)` → `(H, W, K)` float64 response maps
- `gabor_responses(image, ...)` → bank + filter in one call
- `texture_energy(responses, window=9)` → `(H, W, K)` local RMS energy
