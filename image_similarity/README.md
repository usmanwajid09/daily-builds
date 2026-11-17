"""Perceptual-hash image similarity.

Turns images into short hashes (average hash and difference hash) and
compares them with Hamming distance, so near-duplicate images can be
found without comparing every pixel. Numpy only — no OpenCV, no PIL.

## How it works

1. **Shrink** — each image is resized to a small grid (8x8 by default)
   by block averaging, which throws away fine detail and keeps the
   coarse structure.
2. **Hash** — average hash (aHash) records whether each pixel is
   brighter than the image mean; difference hash (dHash) records
   whether each pixel is brighter than its right-hand neighbour.
   Either way the result is a 64-bit integer.
3. **Compare** — the Hamming distance (number of differing bits)
   between two hashes measures dissimilarity: 0 means the hashes are
   identical, small values mean near-duplicates, ~32 is what unrelated
   images score by chance.

## Run the demo

```bash
python image_similarity/demo.py
```

## Use it in your own code

```python
from image_similarity.image_similarity import (
    average_hash, difference_hash, hamming_distance, find_most_similar,
)

h1 = average_hash(image_a)          # 2D grayscale or 3D RGB array
h2 = average_hash(image_b)
print(hamming_distance(h1, h2))     # 0 = identical hash

index, distance = find_most_similar(query, [img1, img2, img3])
print(index, distance)              # index of the closest image
```

## Demo output

```
Perceptual-hash image similarity (64-bit hashes)
query: gradient image, aHash = 0x0f0f0f0f0f0f0f0f, dHash = 0x0000000000000000

candidate        aHash distance  dHash distance
near-duplicate                0               0
flipped                      64              64
random noise                 29              31
solid grey                   32               0

Most similar (aHash): near-duplicate at index 0, Hamming distance 0 / 64
Most similar (dHash): near-duplicate at index 0, Hamming distance 0 / 64

Identical image check: distance of the gradient to itself is 0
```

Notes: the flipped gradient inverts every bit of both hashes (distance
64), because every bright/dark relationship reverses. The solid image
scores 32 against the gradient by chance for aHash, and 0 for dHash —
a flat image and a smooth gradient both hash to all zeros under dHash,
so dHash alone cannot tell them apart; use aHash (or both) when flat
images are in the set.

## Tests

```bash
python -m pytest tests/test_image_similarity.py -q
```
"""
