# LSB Image Steganography

Hide a UTF-8 message in the least significant bits of a uint8 image
(grayscale 2D or RGB 3D) and read it back. Only the lowest bit of each
value is ever touched, so no pixel value changes by more than 1.

## Payload format

```
magic  : b"LSB1"            (4 bytes)
length : message byte count (4 bytes, big-endian)
data   : message bytes     (UTF-8)
```

Bits are written LSB-first across the flattened image.

## Usage

```python
import numpy as np
from steganography_lsb import (
    capacity_bytes, embed_message, extract_message, lsb_difference,
)

image = np.random.default_rng(42).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

print(capacity_bytes(image))          # 1528 message bytes

stego = embed_message(image, "Hello daily-builds!")
print(extract_message(stego))        # 'Hello daily-builds!'
print(lsb_difference(image, stego))  # number of values whose LSB changed
```

`embed_message` raises `ValueError` if the message is too long for the
image, `extract_message` raises `ValueError` if the magic bytes are
wrong/absent, the declared length exceeds capacity, or the payload is
truncated. Non-uint8 images raise `TypeError`; non-str messages raise
`TypeError`.

## Demo

Run from the repo root:

```
python steganography_lsb/demo.py
```

Output (64x64 RGB image, seeded):

```
image shape:       (64, 64, 3) (uint8 RGB)
capacity:          1528 message bytes
embedded message:  'Hello daily-builds!' (19 bytes)
extracted message: 'Hello daily-builds!'
round-trip match:  True
max pixel change:  1 (<= 1: only LSBs changed)
LSBs changed:      110 of 12288 values
shape/dtype kept:  True / True
OK: message hidden and recovered, only LSBs touched.
```

## Tests

```
python -m pytest tests/test_steganography_lsb.py -q
```
