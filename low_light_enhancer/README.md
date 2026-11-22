# low_light_enhancer

Low-light image enhancement from scratch with numpy only. The pipeline is:

1. **Denoise** — a 3×3 median filter, applied *before* brightening so
   sensor noise is not amplified along with the signal.
2. **Adaptive gamma** — the gamma exponent is computed from the image's
   own mean brightness (`gamma = log(target) / log(mean)`, clipped to
   [0.2, 3.0]) so the result lands near a target mean luminance of 0.5.
   Gamma < 1 brightens; a pure-black image is returned unchanged, since
   `0 ** gamma == 0`.
3. **CLAHE** — contrast-limited adaptive histogram equalisation on the
   luminance channel, tile by tile, with histogram clipping so flat
   regions are not over-amplified. Colour is preserved by scaling each
   channel with the luminance ratio.

`metrics()` reports brightness (mean luminance), contrast (standard
deviation) and dynamic range, so the improvement is measurable.

## Run

```bash
python low_light_enhancer/demo.py
```

## Demo output

A synthetic 64×64 dark scene (mean brightness ~0.06) is enhanced.
Expected: brightness rises towards 0.5, contrast and dynamic range both
increase, and the output stays inside [0, 1].

## Tests

```bash
python -m pytest low_light_enhancer/tests/test_low_light_enhancer.py -v
```
