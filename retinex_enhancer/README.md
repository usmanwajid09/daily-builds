# Retinex Enhancer

Single-scale and multi-scale Retinex illumination enhancement,
implemented from scratch with numpy only — no OpenCV.

Retinex models human colour constancy: an image is illumination times
reflectance, so reflectance (the detail we want) is recovered as
`log(image) - log(image blurred with a Gaussian)`. Single-scale Retinex
(SSR) uses one blur scale; multi-scale Retinex (MSR) averages several;
MSRCR adds colour restoration so colours do not wash out. Useful for
dark, unevenly lit or backlit photos.

## Usage

```python
import numpy as np
from retinex_enhancer import enhance, single_scale_retinex, multi_scale_retinex

image = np.load("dark.npy")  # 2D grayscale or (H, W, 3) RGB, values in [0, 255]
bright = enhance(image, method="msr")           # uint8 array in [0, 255]
ssr = single_scale_retinex(image, sigma=80.0)   # float array in [0, 255]
msr = multi_scale_retinex(image, sigmas=(15.0, 80.0, 250.0))
msrcr = multi_scale_retinex_cr(image)
```

`enhance(image, method="ssr" | "msr" | "msrcr")` returns a uint8 image
and forwards extra keyword arguments to the chosen method. Results
are min-max stretched to [0, 255]; a flat image maps to constant
mid-grey (128), since it contains no reflectance detail. Images must
be 2D or 3-channel RGB with values in [0, 255]; sigmas must be
positive; anything else raises `ValueError`.

## Run the demo

```bash
PYTHONDONTWRITEBYTECODE=1 python retinex_enhancer/demo.py
```

Builds a dark, unevenly lit synthetic image and prints mean/std
before and after SSR, MSR and MSRCR.

## Run the tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_retinex_enhancer.py -q
```
