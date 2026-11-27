"""Demo: Floyd-Steinberg dithering of a synthetic gradient.

Run from the repo root:  python floyd_steinberg_dithering/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from floyd_steinberg_dithering import dither_color, dither_grayscale


def build_gradient(height=40, width=80):
    """Smooth horizontal gradient plus a soft radial disc."""
    gradient = np.tile(np.linspace(0, 255, width), (height, 1))
    yy, xx = np.mgrid[0:height, 0:width]
    disc = np.clip(255 - 6 * np.sqrt((yy - 20) ** 2 + (xx - 60) ** 2),
                   0, 255)
    return np.clip((gradient + disc) / 2, 0, 255)


def main():
    gray = build_gradient()
    dithered = dither_grayscale(gray, serpentine=True)
    plain = dither_grayscale(gray, serpentine=False)

    print(f"Gradient shape: {gray.shape}, mean brightness: {gray.mean():.1f}")
    print(f"Dithered values: {sorted(np.unique(dithered).tolist())}")
    print(f"Dithered mean brightness: {dithered.mean():.1f} "
          f"(input mean {gray.mean():.1f})")
    print(f"White-pixel fraction: {float((dithered == 255).mean()):.3f} "
          f"(input mean / 255 = {gray.mean() / 255:.3f})")
    print(f"Serpentine vs plain differ at "
          f"{int((dithered != plain).sum())} pixels")

    rgb = np.stack([gray, np.flip(gray, axis=1),
                    np.full_like(gray, 128.0)], axis=-1).astype(np.uint8)
    colour = dither_color(rgb, levels=2)
    print(f"Colour dithered shape: {colour.shape}, dtype: {colour.dtype}")
    print(f"Distinct colours after colour dither: "
          f"{len(np.unique(colour.reshape(-1, 3), axis=0))} (max 8)")


if __name__ == "__main__":
    main()
