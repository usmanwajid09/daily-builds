# Barcode Decoder

Decode a simple 1D barcode from a synthetic image (numpy only).

Text is encoded as bars, rendered to a grayscale image, and decoded
back to text — the round-trip is exact for every character in the
alphabet.

## Encoding scheme

- **Alphabet:** `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ` (37 characters:
  digits, uppercase letters, space).
- **Character pattern:** each character maps to a fixed pattern of
  **6 bars**, alternating black/white starting with black. The pattern
  is the 6-bit binary form of the character's index in the alphabet
  (most-significant bit first): a `0` bit is a narrow bar (1 unit),
  a `1` bit is a wide bar (2 units).
- **Why 6 bars, not 5:** 5 bars give only 2**5 = 32 patterns, which
  cannot cover 37 characters plus two sentinels (39 distinct patterns).
  Six bars give 64 patterns, so every character and both sentinels
  have a unique pattern.
- **Sentinels** (also 6 bars, black first): start `111111` (index 63),
  stop `101010` (index 42) — both outside the alphabet index range
  0–36, so neither collides with a character.
- **Layout (in units):** quiet zone (10 white) + start + one 6-bar
  group per character + stop + quiet zone (10 white). The stop
  pattern's last bar is white, so it merges with the trailing quiet
  zone into a single white run of 11 units.
- **Rendering:** each unit is `unit_width` pixels wide (default 3),
  the image is `height` rows tall (default 40), black bars are 0 and
  white is 255 (uint8 grayscale).

## Decoding

Scans the middle row, thresholds at 128, run-length encodes the row,
infers the unit width from the shortest interior run, converts each
run to 1 or 2 units, then maps each 6-bar group back to a character.
Raises `ValueError` on a missing/invalid start or stop sentinel, an
unknown character pattern, or run lengths that do not map to 1 or 2
units.

## Usage

```python
from barcode_decoder import encode, render_barcode, decode, barcode_width

image = render_barcode("HELLO 42")   # (40, 318) uint8 array, values 0/255
print(decode(image))                 # 'HELLO 42'
print(barcode_width("HELLO 42"))     # 318
print(encode("A"))                   # run widths in units, white first
```

## How to run

```bash
python barcode_decoder/demo.py     # runs from any working directory
python -m pytest tests/test_barcode_decoder.py -q
```

## Demo result

```
text:    'HELLO 42'
shape:   (40, 318) (height x width)
width:   318 px (barcode_width: 318 px)
decoded: 'HELLO 42'
match:   True
second:  'ABC123' -> shape (40, 264), decoded 'ABC123', match: True
```

"HELLO 42" renders to a 40×318 image (106 units × 3 px) and decodes
back exactly; "ABC123" renders to 40×264 (88 units × 3 px) and also
round-trips exactly.

## Tests

13 pytest tests in `tests/test_barcode_decoder.py`: round-trips for
"HELLO", "ABC123", "9", and "A B"; render shape/width matching
`barcode_width`; rendered image contains only 0 and 255; decoding an
all-white image raises `ValueError`; decoding an image with a
corrupted start sentinel raises `ValueError`; `encode` rejects
lowercase/unsupported characters and the empty string; invalid
`unit_width`/`height` raise `ValueError`; `decode` rejects 3D and
non-array input; exact round-trip for every single character in the
alphabet; round-trip still works with `unit_width=1`.
