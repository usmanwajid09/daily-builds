# Digit Augmentation

Image augmentation pipeline for grayscale digit images (e.g. 28x28
MNIST-style arrays), numpy only: 90-degree rotations, horizontal and
vertical flips, integer-pixel translations (zero-filled), and Gaussian
noise. `augment` applies a random combination of them, and
`augment_n` returns `n` different augmented copies of one image —
the usual way to expand a small digit dataset before training.

## Usage

```bash
python digit_augmentation/demo.py
```

```python
from digit_augmentation import augment, augment_n, rotate90, translate

copies = augment_n(image, n=5, seed=42)   # list of 5 augmented images
one = augment(image, seed=1)              # one random combination
rotated = rotate90(image, k=1)            # 90 degrees counter-clockwise
shifted = translate(image, dy=2, dx=-1)   # rolled-in area filled with 0
```

Noise is clipped to the image's own value range (`[min, max]`, or
`[0, 1]` when the image lies in `[0, 1]`). Bad inputs — non-2D or
empty arrays, non-finite values, negative `std`, non-positive `n`,
non-integer `k` — raise `ValueError` with a clear message.

## Demo

The demo builds a synthetic 28x28 digit "1" (bright vertical bar,
cols 13-14, rows 4-23, plus a small diagonal stroke) and runs
`augment_n` with `n=5`, `seed=42`:

```text
original: shape=(28, 28), sum=45.0
augmented 0: shape=(28, 28), sum=62.77, differs from original: True
augmented 1: shape=(28, 28), sum=46.01, differs from original: True
augmented 2: shape=(28, 28), sum=66.21, differs from original: True
augmented 3: shape=(28, 28), sum=47.11, differs from original: True
augmented 4: shape=(28, 28), sum=54.69, differs from original: True
rotate90 k=1:       sum=45.0
flip_horizontal:    sum=45.0
flip_vertical:      sum=45.0
translate (2, 3):   sum=45.0
noise std=0.1 s=7:  sum=67.26
```

Rotations, flips and translations preserve the pixel sum (they only
move pixels); the noise example adds ~22 to the sum because
clipping at 0 pushes the zero background up while the bright pixels
stay near 1.

## Run tests

```bash
python -m pytest digit_augmentation/tests/test_digit_augmentation.py -q
```

13 tests: rotations match `np.rot90` (and 4 rotations return the
original), flips match `np.fliplr`/`np.flipud`, translation identity
/ pixel shift / zero fill, noise identity at `std=0` and
reproducibility with a fixed seed, shape preservation, `augment_n`
count/shape/reproducibility, and `ValueError` on invalid inputs.
