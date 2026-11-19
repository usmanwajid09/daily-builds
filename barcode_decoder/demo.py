"""Demo: encode text as a 1D barcode, render it, decode it back.

Encodes ``DEMO_TEXT`` ("HELLO 42"), renders the barcode image, prints
its shape and width, decodes it, and checks the round-trip. A second
round-trip ("ABC123") is shown as well.
"""

import os
import sys

# Allow running as a plain script (`python barcode_decoder/demo.py`):
# make the repo root importable so `barcode_decoder` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from barcode_decoder.barcode import barcode_width, decode, render_barcode

DEMO_TEXT = "HELLO 42"
SECOND_TEXT = "ABC123"


def main():
    image = render_barcode(DEMO_TEXT)
    decoded = decode(image)
    print(f"text:    {DEMO_TEXT!r}")
    print(f"shape:   {image.shape} (height x width)")
    print(f"width:   {image.shape[1]} px "
          f"(barcode_width: {barcode_width(DEMO_TEXT)} px)")
    print(f"decoded: {decoded!r}")
    print(f"match:   {decoded == DEMO_TEXT}")

    second_image = render_barcode(SECOND_TEXT)
    second_decoded = decode(second_image)
    print(f"second:  {SECOND_TEXT!r} -> shape {second_image.shape}, "
          f"decoded {second_decoded!r}, match: {second_decoded == SECOND_TEXT}")
    return decoded


if __name__ == "__main__":
    main()
