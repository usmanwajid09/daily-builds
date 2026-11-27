# Floyd–Steinberg Dithering

Converts a grayscale image to black-and-white (or an RGB image to a
small palette) with Floyd–Steinberg error-diffusion dithering.

## How it works

Pixels are visited in scan order. Each pixel is quantized to the
nearest palette value (0 or 255 for binary), and the quantization
error is pushed forward to the not-yet-visited neighbours with the
classic weights — right 7/16, down-left 3/16, down 5/16, down-right
1/16 — so the local average brightness is preserved and smooth
gradients render as dot patterns instead of hard bands. Serpentine
scanning (alternating row direction) reduces directional streaking.

`dither_color` diffuses the error independently per channel, so with
`levels=2` the output uses at most the 8 corners of the RGB cube.

## Run

```bash
python floyd_steinberg_dithering/demo.py
python -m pytest tests/test_floyd_steinberg_dithering.py -q
```

## Example output

```
Gradient shape: (40, 80), mean brightness: 112.0
Dithered values: [0, 255]
Dithered mean brightness: 111.9 (input mean 112.0)
White-pixel fraction: 0.439 (input mean / 255 = 0.439)
Serpentine vs plain differ at 977 pixels
Colour dithered shape: (40, 80, 3), dtype: uint8
Distinct colours after colour dither: 8 (max 8)
```
