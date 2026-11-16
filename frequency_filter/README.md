# Frequency Filter (FFT)

FFT-based low-pass / high-pass filtering — no OpenCV, just numpy. It
moves an image into the frequency domain, multiplies by a mask that
keeps or drops frequency bands, and transforms back to pixels.

## How it works

1. `np.fft.fft2` turns the image into its frequency spectrum, and
   `fftshift` moves the zero frequency (DC) to the centre.
2. A mask is built over that centred spectrum: values near 1 keep a
   frequency, values near 0 kill it. The mask's distance from the
   centre is the frequency — smooth gradients live near the centre,
   fine detail, edges and noise live far out.
3. Multiply spectrum by mask, `ifft2`, and take the real part.

**Low-pass** keeps the centre of the spectrum: smooth structure stays,
fine detail and noise go — a blur that is sharp in frequency terms.
**High-pass** keeps everything *except* the centre: the smooth
background and the image's mean disappear, leaving edges and detail.

## Ideal vs Gaussian

- **Ideal** — a hard disc: 1 inside the cutoff radius, 0 outside. It
  cuts frequencies exactly, but the sharp edge causes *ringing*
  (ripples) around edges in the output.
- **Gaussian** — a smooth falloff, `exp(-r² / 2·cutoff²)`. No ringing,
  at the price of a softer cutoff. Usually the better default.

The cutoff is a radius in frequency bins: a small cutoff keeps only
the very smoothest content; a larger cutoff keeps more detail.

## Usage

```bash
pip install -r requirements.txt
python -m frequency_filter.demo
```

The demo builds a synthetic 128x128 image (smooth gradient + fine
checkerboard + noise) and prints statistics before and after Gaussian
low-pass and high-pass filtering, including the fraction of energy the
low-pass keeps.

Use it on your own image from Python:

```python
import numpy as np
from frequency_filter.filter import low_pass, high_pass, make_mask

image = np.load("image.npy")          # 2D grayscale, or 3D RGB (filtered per channel)
smooth = low_pass(image, cutoff=8.0, kind="gaussian")
edges = high_pass(image, cutoff=8.0, kind="gaussian")
mask = make_mask(image.shape, cutoff=8.0, kind="ideal")  # inspect the mask itself
```

## Tests

```bash
python -m pytest tests/test_frequency_filter.py -q
```
