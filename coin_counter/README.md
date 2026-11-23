# Coin Counter

Count coins in a synthetic image and total their value, using only
numpy -- no OpenCV, no scipy.

## How it works

1. **Generate** a synthetic scene with `make_coins_image`: a dark
   background with bright filled circles (plus a little Gaussian
   noise), so no external image files are needed.
2. **Threshold** the image to a binary mask (bright = coin).
3. **Distance transform**: a two-pass chamfer (3-4) transform gives
   every foreground pixel its approximate distance to the nearest
   background pixel. A coin's centre is the peak of that transform,
   and the peak value is (approximately) the coin's radius.
4. **Local maxima + non-maximum suppression** find the coin centres:
   a pixel is a candidate if it is >= every neighbour in its 3x3
   window and >= `min_radius`; candidates within half the radius of
   a stronger accepted coin are dropped (this merges the small
   plateau of equal maxima at one coin's centre).
5. **Classify by radius**: each detected radius is matched to the
   nearest known radius in the denomination map.

## Denomination map

| Radius (px) | Value |
|-------------|-------|
| 10          | 1     |
| 14          | 5     |
| 18          | 10    |

A detected radius more than 3 px from every known radius is still
counted, but gets denomination `None` and contributes 0 to the
total value.

## Usage

```python
from coin_counter.coins import make_coins_image, count_coins

image = make_coins_image([(40, 40, 10), (130, 120, 14), (165, 155, 18)],
                         size=200, seed=7)
coins, summary = count_coins(image)
print(summary)
# {'count': 3, 'total_value': 16, 'by_denomination': {1: 1, 5: 1, 10: 1}}
```

Coins can also be given by denomination instead of radius:

```python
image = make_coins_image([(50, 50, 5), (150, 150, 10)],
                         by="denomination")
```

`count_coins` accepts 2D grayscale or 3D RGB(A) images, normalised
(0-1) or 0-255, and validates its inputs (`threshold` in (0, 1) or
(0, 255) depending on the image scale, `min_radius >= 1`).

## Demo

Run `python coin_counter/demo.py` from the repo root. It builds a
200x200 scene with 3x radius-10, 2x radius-14 and 1x radius-18 coins
and counts them:

```
image: 200x200 grayscale, seed=7
ground truth: 6 coins, total value 23
detected:     6 coins, total value 23
  coin at (100, 35), radius 10.3, denomination 1
  coin at (40, 40), radius 10.3, denomination 1
  coin at (160, 45), radius 10.3, denomination 1
  coin at (130, 120), radius 13.3, denomination 5
  coin at (50, 130), radius 13.3, denomination 5
  coin at (165, 155), radius 17.3, denomination 10
per-denomination breakdown (detected): {1: 3, 5: 2, 10: 1}
per-denomination breakdown (truth): {1: 3, 5: 2, 10: 1}
exact match with ground truth: True
```

Detected radii run ~0.7 px under the true radius (the chamfer
approximation slightly underestimates diagonal distances); the
nearest-radius match absorbs that bias. Slightly overlapping coins
are still separated because their distance-transform peaks stay
distinct; heavily overlapping coins may merge into one peak, which
is a known limitation of this approach.

## Tests

`python -m pytest tests/test_coin_counter.py -q` — 26 tests covering
image generation, the distance transform, exact counts and totals,
RGB/normalised inputs, slightly overlapping coins, noise, unknown
radii, and invalid inputs.
