# Privacy Blur (YCbCr skin detection + box blur)

Detect skin-coloured regions in an RGB image with the classic
rule-based YCbCr test — Cb in [77, 127] and Cr in [133, 173] (Chai &
Ngan bounds), the same rule as `skin_detector` — and blur them for
privacy. Masked pixels are replaced by a box-blurred version of the
image (padded mean filter via an integral image, so large kernels are
cheap); every non-skin pixel is returned exactly unchanged. Numpy
only, no new dependencies.

## Why

Blurring skin regions is a cheap first step for anonymising photos
and video frames — faces, hands, and other identifiable skin areas
become unrecognisable without touching the rest of the scene.

## Usage

```bash
python privacy_blur/demo.py
```

```python
from privacy_blur import privacy_blur, blur_stats, detect_skin

blurred, mask = privacy_blur(image)   # blurred image + boolean skin mask
print(blur_stats(image))              # skin_pixels, skin_fraction, blurred_pixels
```

## Demo result

The demo builds a synthetic 120x160 scene (dark-blue background,
green circle, one skin-coloured rectangle RGB 210, 170, 140 at rows
40-89, cols 50-109) and blurs it with the default 5x5 kernel:

```
scene: 120x160, skin rectangle rows 40-89, cols 50-109
skin fraction: 0.1562 (expected 0.1562)
skin pixels:   3000
blurred pixels (value changed): 424
inside rectangle changed: True
outside rectangle unchanged: True
```

The skin fraction matches the rectangle exactly (3000 / 19200).
Only 424 of the 3000 skin pixels change value: the rectangle is
uniform, so its interior blurs to itself and only the edge pixels —
where the blur mixes in background colour — actually change.

## Run tests

```bash
python -m pytest privacy_blur/tests/test_privacy_blur.py -v
```

12 tests: solid-skin masking, blue/green not skin, YCbCr of pure red,
blur changing skin but not non-skin pixels, kernel size 1 leaving the
image unchanged, invalid inputs raising `ValueError`, and scene stats
matching the known rectangle.
