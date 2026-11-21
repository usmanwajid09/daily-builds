# color_constancy

Colour constancy / white balance from scratch (numpy only). Estimates
the colour of the scene illuminant and removes the cast by scaling
each channel.

## Estimators

- **gray-world** — the average scene colour is grey: scale each
  channel so all three channel means are equal.
- **white-patch** — the brightest pixels are white: scale each channel
  so its 99th percentile maps to 255.
- **shades-of-gray** — a p-norm (Minkowski) generalisation; p = 1 is
  gray-world, larger p approaches white-patch.

## API

- `gray_world_gains(image)`, `white_patch_gains(image, percentile=99)`,
  `shades_of_gray_gains(image, p=6)` — per-channel gains (shape (3,)).
- `estimate_illuminant(image, method="gray_world", ...)` — normalised
  RGB illuminant triplet (sums to 1).
- `apply_gains(image, gains)` — multiply each channel by the gains.
- `white_balance(image, method="gray_world", ...)` — corrected image.

Images may be uint8 or float in [0, 1] / [0, 255]; the result comes
back in the same dtype and range. Grayscale / non-RGB input raises
`ValueError`.

## Run

```bash
python -m pytest tests/test_color_constancy.py -q
python color_constancy/demo.py
```

## Example output

```
original scene means (R,G,B): [109.5 111.9 123.7]
tinted image means (R,G,B):   [157.9 111.9  67.6]
true cast applied:            [1.45 1.   0.55]
gray_world      corrected means: [112.5 112.5 112.5] (spread  0.03), estimated illuminant [0.468 0.332 0.2  ], mean |error| vs scene: 4.96
white_patch     corrected means: [157.9 155.  170.5] (spread 15.53), estimated illuminant [0.472 0.341 0.187], mean |error| vs scene: 46.12
shades_of_gray  corrected means: [109.5 112.8 119.7] (spread 10.24), estimated illuminant [0.481 0.331 0.188], mean |error| vs scene: 1.79
```
