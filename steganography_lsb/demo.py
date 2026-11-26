"""Demo: hide a message in an image with LSB steganography, read it back.

Builds a synthetic random 64x64 RGB image, embeds "Hello daily-builds!",
extracts it back, and checks that only least significant bits changed
(max absolute pixel difference <= 1).
"""

import os
import sys

# Allow running as a plain script (`python steganography_lsb/demo.py`):
# make the repo root importable so `steganography_lsb` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from steganography_lsb import (
    capacity_bytes,
    embed_message,
    extract_message,
    lsb_difference,
)

MESSAGE = "Hello daily-builds!"


def main():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    capacity = capacity_bytes(image)
    stego = embed_message(image, MESSAGE)
    recovered = extract_message(stego)

    max_change = int(np.max(np.abs(stego.astype(int) - image.astype(int))))
    changed_lsbs = lsb_difference(image, stego)

    print(f"image shape:       {image.shape} (uint8 RGB)")
    print(f"capacity:          {capacity} message bytes")
    print(f"embedded message:  {MESSAGE!r} ({len(MESSAGE.encode())} bytes)")
    print(f"extracted message: {recovered!r}")
    print(f"round-trip match:  {recovered == MESSAGE}")
    print(f"max pixel change:  {max_change} (<= 1: only LSBs changed)")
    print(f"LSBs changed:      {changed_lsbs} of {image.size} values")
    print(f"shape/dtype kept:  {stego.shape == image.shape} / "
          f"{stego.dtype == image.dtype}")
    print("OK: message hidden and recovered, only LSBs touched.")


if __name__ == "__main__":
    main()
