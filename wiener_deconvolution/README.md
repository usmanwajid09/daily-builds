# Wiener Deconvolution

FFT-based image deblurring from scratch (numpy only). A blurred,
noisy image `g = h * f + n` is restored in the frequency domain with
the Wiener filter `F = conj(H) / (|H|^2 + NSR) * G`. Where the blur
attenuates a frequency strongly the filter backs off instead of
amplifying the noise there, which is what plain inverse filtering
would do.

## Run

```bash
python wiener_deconvolution/demo.py
python -m pytest tests/test_wiener_deconvolution.py -q
```

## Example

```python
from wiener_deconvolution import make_gaussian_psf, wiener_deconvolve, estimate_nsr

psf = make_gaussian_psf(size=15, sigma=2.0)
nsr = estimate_nsr(noise_var=2.25, signal_var=float(np.var(image)))
restored = wiener_deconvolve(blurred, psf, nsr=nsr)
```

Demo on a synthetic 64×64 image, Gaussian blur (sigma 2) + noise (sigma 1.5):

```
image: 64x64, gaussian PSF sigma=2.0, added noise sigma=1.5
estimated NSR: 0.000588
PSNR blurred+noisy: 20.63 dB
PSNR Wiener restored: 22.99 dB (gain +2.36 dB)
PSNR noise-free inverse-filtered: 131.82 dB
```

## API

- `make_gaussian_psf(size=15, sigma=2.0)` → centred, normalised PSF
- `make_motion_psf(length=15, angle_degrees=0.0)` → straight-line motion-blur PSF
- `apply_blur(image, psf)` → blurred image (circular convolution via FFT)
- `wiener_deconvolve(blurred, psf, nsr=0.01)` → restored image (not clipped)
- `estimate_nsr(noise_var, signal_var)` → noise-to-signal power ratio
- `psnr(original, restored, data_range=None)` → dB (inf for identical images)
