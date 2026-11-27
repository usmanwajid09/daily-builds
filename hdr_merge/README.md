# HDR Merge

Combines several exposures of the same scene into one high-dynamic-range
radiance map (Debevec & Malik style, linear response assumed) and tone
maps it back to a displayable 8-bit image with the Reinhard operator.

## How it works

1. **Hat weight** (`hat_weight`): each pixel value is weighted by
   min(z, 255 - z) — zero for saturated (0 / 255) pixels, maximal at
   mid-grey — so blown-out and crushed pixels never contribute.
2. **Merge** (`merge_hdr`): per pixel, log-radiance is the weighted
   average of (ln z - ln exposure_time) over the exposures. Pixels
   saturated in every exposure fall back to the unweighted mean.
3. **Tone map** (`reinhard_tone_map`): the global Reinhard operator
   Ld = L (1 + L / Lw^2) / (1 + L), normalised so the white point maps
   to 1.0, applied via the luminance curve so colours are preserved,
   with optional gamma.

## Run

```bash
python hdr_merge/demo.py
python -m pytest tests/test_hdr_merge.py -q
```

## Example output

```
Scene shape: (48, 64, 3), radiance range [0.120, 3.000]
Exposures: 4 at times [0.25, 1.0, 4.0, 16.0]
Hat weights at 0 / 127.5 / 255: 0.0 / 127.5 / 0.0
HDR radiance range: [0.1250, 3.01]
HDR vs true scene (log domain) correlation: 0.9988
Tone-mapped shape: (48, 64, 3), dtype: uint8
Tone-mapped range: [28, 255]
PSNR tone-mapped vs mid exposure: 11.47 dB
```
