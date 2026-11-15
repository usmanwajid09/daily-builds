# Colour Transfer

Reinhard colour transfer from scratch (numpy only).

Re-colours a source image so its colour statistics match a target
image: for each channel, subtract the source mean, divide by the
source standard deviation, then scale by the target standard
deviation and add the target mean. By default this is done in CIELAB
space (implemented with the standard sRGB/D65 matrices), where the
channels are roughly decorrelated; plain RGB transfer is also
available via `transfer_rgb`.

## Run

```bash
python color_transfer/demo.py
python -m pytest tests/test_color_transfer.py -q
```

## Example

Demo transferring a warm gradient source onto a cool gradient target
(Lab space — the space the transfer matches, so Lab statistics are
the ones that should agree):

```
source   Lab mean: [63.4, 28.3, 47.8]  Lab std: [2.7, 3.5, 3.0]
target   Lab mean: [53.3, 12.5, -54.4]  Lab std: [3.3, 3.8, 4.7]
result   Lab mean: [53.4, 12.7, -54.3]  Lab std: [3.2, 3.6, 4.5]
```

The result's Lab statistics closely match the target's.

## API

- `color_transfer(source, target, colorspace="lab")` → RGB image in
  [0, 255], same shape as `source`
- `transfer_rgb(source, target)` → same, done directly in RGB space
- `rgb_to_lab(rgb)` / `lab_to_rgb(lab)` → colour-space conversions
