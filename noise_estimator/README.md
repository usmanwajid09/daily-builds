# Noise Estimator

Single-image Gaussian noise estimation (numpy only) plus a small
denoising guide. Two Laplacian-based estimators of the noise
standard deviation `sigma`:

- **immerkaer** (Immerkaer, 1996): `sqrt(pi/2) * mean(|Laplacian|) / 6`
- **mad**: a robust variant, `median(|response|) / 0.6745 / 6`, less
  sensitive to the strong responses real edges produce

The Immerkaer mask is near zero on smooth content while white noise
of std `sigma` gives a response of std `6 * sigma`, so no clean
reference image is needed.
`recommend_denoising` turns the estimate into a concrete suggestion
(method + parameters), and `gaussian_denoise` applies the light
Gaussian blur recommended for low-to-moderate noise.

## Run

```bash
python noise_estimator/demo.py
python -m pytest tests/test_noise_estimator.py -q
```

## Example

```python
from noise_estimator import estimate_sigma, recommend_denoising

sigma = estimate_sigma(noisy_image)          # e.g. 7.5
rec = recommend_denoising(sigma)
# {"level": "moderate", "method": "gaussian_blur",
#  "parameters": {"sigma": 1.2}, "note": "A Gaussian blur ..."}
```

Demo on a synthetic smooth image with true sigma = 12:

```
true noise sigma: 12.0
estimated sigma (immerkaer): 12.12 (error 0.12)
estimated sigma (mad): 11.91 (error 0.09)
estimated sigma on the clean image: 0.00
noise level: high
recommendation: bilateral_filter {'diameter': 9, 'sigma_color': 24.24, 'sigma_space': 2.0} — Noise is strong; prefer an edge-preserving bilateral filter over a plain blur.
PSNR noisy: 25.73 dB, PSNR after denoise: 37.55 dB (gain +11.83 dB)
```

## API

- `estimate_sigma(image, method="immerkaer")` → float; `method` is `"immerkaer"` or `"mad"`
- `noise_level(sigma)` → `"very low" | "low" | "moderate" | "high" | "very high"` (0..255 scale)
- `recommend_denoising(sigma)` → `{"level", "method", "parameters", "note"}`
- `gaussian_denoise(image, sigma=1.0)` → blurred image
- `psnr(original, restored, data_range=None)` → dB (inf for identical images)
