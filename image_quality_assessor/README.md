# Image Quality Assessor (no-reference)

Score an image's quality without needing a pristine original to compare
against. Works on grayscale and RGB images (numpy arrays, 0-255 or 0-1)
and returns a 0-100 score with a verdict: `excellent` (>= 80), `good`
(>= 60), `fair` (>= 40), `poor` (< 40).

## How it works

Four measurements, combined with weights blur 35%, noise 30%,
contrast 20%, brightness 15%:

- **Blur** — variance of the Laplacian response, using a 3x3 Laplacian
  kernel convolved with numpy (no OpenCV). Sharp images have strong
  second derivatives at edges. The sub-score is computed on a lightly
  box-blurred copy, so sensor noise cannot masquerade as sharpness.
- **Noise** — estimated Gaussian sigma (grey levels) from the high-pass
  residual (image minus a 3x3 box blur) inside the centre of 8x8 blocks,
  median across blocks, so real edges don't inflate the estimate.
- **Contrast** — standard deviation of the pixel values.
- **Brightness** — mean pixel value; mid-range brightness scores best,
  near-black / near-white score worst.

## Usage

```python
from image_quality_assessor import assess, laplacian_variance, estimate_noise

result = assess(image)  # image: 2D grayscale or (H, W, 3/4) RGB(A), 0-255 or 0-1
# {'score': 88.7, 'verdict': 'excellent', 'blur': 16989.1, 'noise': 0.0,
#  'contrast': 92.0, 'brightness': 127.5, 'blur_score': ..., ...}

laplacian_variance(image)  # classic sharpness measure
estimate_noise(image)      # estimated noise sigma in grey levels
```

Invalid inputs (wrong shape, empty, NaN/inf, values outside 0-255 or
0-1, non-numeric dtype) raise `ValueError` with a clear message.

## Demo

```bash
python image_quality_assessor/demo.py
```

The demo assesses a synthetic sharp checkerboard and degraded versions
of it (all generated in code, no external files):

| image | score | verdict |
| --- | --- | --- |
| sharp | 88.7 | excellent |
| blurred (box blur x12) | 59.0 | fair |
| noisy (sigma=30) | 67.9 | good |
| low contrast | 53.0 | fair |
| dark (x0.12) | 40.4 | fair |

Sharp scores highest; every degradation drops the score, as expected.

## Run tests

```bash
python -m pytest tests/test_image_quality_assessor.py -q
```

## Limitations

The score assumes textured, natural-looking content: a perfectly smooth
gradient has almost no Laplacian response and scores low on blur even
though it is a "clean" image. The noise estimate assumes roughly
Gaussian noise and mostly flat regions; fine texture everywhere
(e.g. 1-pixel checkerboards) reads as noise.
