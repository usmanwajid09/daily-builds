# Image Registration

Translation registration via phase correlation (numpy only).

Phase correlation estimates the integer (optionally subpixel) shift
between two images from the phase of their cross-power spectrum:
the inverse FFT of the normalised cross-power spectrum peaks exactly
at the translation between the images. Parabolic interpolation
refines the peak to subpixel accuracy, and fractional shifts use the
Fourier shift theorem (with the Nyquist bin handled at its
bandlimited, real value).

## Run

```bash
python image_registration/demo.py
python -m pytest tests/test_image_registration.py -q
```

## Example

```python
from image_registration import estimate_shift, register_translation

dy, dx = estimate_shift(reference, moving)          # (dy, dx) shift
result = register_translation(reference, moving, subpixel=True)
# result: {"shift": (dy, dx), "registered": array, "peak": float}
```

Demo on a synthetic 64×64 white-noise pattern:

```
integer shift: true=(5, -7), estimated=(5.0, -7.0), error=0.000 px, peak=1.000
subpixel shift: true=(2.5, -3.5), estimated=(2.48, -3.47), error=0.031 px, peak=0.417
registered image matches reference: max abs diff = 0.00e+00
```

Integer shifts are recovered exactly with a peak of 1. The subpixel
estimate lands within 0.03 px; its peak is lower (0.42) because a
fractional shift spreads the correlation peak over neighbouring
pixels — that spread is exactly what the parabolic fit measures.## API

- `phase_correlation(reference, moving, subpixel=False)` → `((dy, dx), peak)`
- `estimate_shift(reference, moving, subpixel=False)` → `(dy, dx)` only
- `apply_shift(image, dy, dx)` → image rolled by `(dy, dx)`; exact for integer shifts, Fourier shift for fractional ones
- `register_translation(reference, moving, subpixel=False)` → dict with `shift`, `registered` image and correlation `peak`
