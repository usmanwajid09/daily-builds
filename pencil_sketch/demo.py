"""Demo: pencil-sketch effect on a synthetic image.

Builds a 60x80 image — a horizontal gradient with a dark disc and a
dark rectangle on it — runs :func:`pencil_sketch.pencil_sketch` and
prints the output range, the mean of a flat bright region (should stay
near white) and the mean along an edge (should be dark).
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pencil_sketch.model import pencil_sketch


def make_image(height=60, width=80):
    """Gradient (bright top-left to mid-grey) + dark disc + dark rectangle."""
    y, x = np.mgrid[0:height, 0:width]
    image = 230.0 - 0.8 * x - 0.5 * y  # 230 down to ~140
    disc = (x - 25) ** 2 + (y - 30) ** 2 <= 10 ** 2
    image[disc] = 30.0
    image[38:50, 55:75] = 60.0  # dark rectangle
    return np.clip(image, 0, 255).astype(np.float64)


def main():
    image = make_image()
    sketch = pencil_sketch(image, sigma=1.0)

    # Flat bright region: top-left corner, away from any edge.
    flat = sketch[2:10, 2:10]
    # Edge: a ring just *inside* the disc boundary (radius 8..10). In a
    # pencil sketch the dark line sits on the dark side of an edge: just
    # inside the disc the base is dark (30) while the blurred inverted
    # image is still brightened by the bright surroundings.
    y, x = np.mgrid[0:60, 0:80]
    r = np.sqrt((x - 25) ** 2 + (y - 30) ** 2)
    edge = sketch[(r >= 8) & (r <= 10)]

    print(f"input shape: {image.shape}, output shape: {sketch.shape}, dtype: {sketch.dtype}")
    print(f"output min: {int(sketch.min())}, max: {int(sketch.max())}")
    print(f"flat bright region mean: {flat.mean():.1f} (should stay near white, > 200)")
    print(f"edge mean: {edge.mean():.1f} (should be dark, clearly below the flat region)")
    print(f"overall mean: {sketch.mean():.1f}")


if __name__ == "__main__":
    main()
