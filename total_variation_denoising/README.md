# Total-Variation Denoising

Denoises a grayscale image with the Chambolle projection algorithm,
which minimises total variation plus a data-fidelity term. Noise is
removed while sharp edges are preserved — unlike a Gaussian blur,
which smears edges.

## How it works

Chambolle (2004) solves the dual problem with projected gradient
ascent on a dual vector field `p`:

```
p <- (p + tau * grad(div p - f / weight)) / (1 + tau * |grad(div p - f / weight)|)
u  = f - weight * div p
```

with `tau = 0.125` (the 2D stability limit). `weight` is the denoising
weight: larger values smooth more aggressively, smaller values stay
closer to the noisy input. Iteration stops when the
max pixel change drops below `tol` or `max_iter` is reached.

## Run

```bash
python total_variation_denoising/demo.py
python -m pytest tests/test_total_variation_denoising.py -q
```

## Example output

```
Image shape: (48, 64)
PSNR noisy vs clean:    24.64 dB
PSNR TV-denoised:       36.88 dB
Denoised dtype: float64, range [28.4, 208.4]
Block mean: 200.3, background mean: 90.3
```
