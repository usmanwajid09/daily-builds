"""Demo: resize a synthetic gradient image up and down.

Builds a 40x60 RGB gradient image (red grows left to right, green top
to bottom, blue constant), resizes it up to 80x120 and down to 10x15
with both nearest-neighbour and bilinear interpolation, and prints the
shapes plus a sample pixel check at the centre.
"""

import numpy as np

from image_resize.resize import resize


def make_gradient(height=40, width=60):
    """Return an RGB image: R ramps across, G ramps down, B is 128."""
    rows = np.linspace(0, 255, height)[:, None]
    cols = np.linspace(0, 255, width)[None, :]
    r = np.broadcast_to(cols, (height, width))
    g = np.broadcast_to(rows, (height, width))
    b = np.full((height, width), 128.0)
    return np.stack([r, g, b], axis=2)


def main():
    image = make_gradient()
    print(f"original: {image.shape}")

    for method in ("nearest", "bilinear"):
        up = resize(image, 80, 120, method=method)
        down = resize(image, 10, 15, method=method)
        centre = tuple(np.round(up[40, 60]).astype(int).tolist())
        print(f"{method:>8}: up {up.shape}, down {down.shape}, "
              f"centre pixel of upscaled image: {centre}")

    # A linear gradient is reproduced exactly by bilinear interpolation.
    grad = np.linspace(0, 255, 60)[None, :].repeat(40, axis=0)
    back = resize(resize(grad, 80, 120, method="bilinear"), 40, 60,
                  method="bilinear")
    max_err = float(np.max(np.abs(back - grad)))
    print(f"bilinear round-trip max error on a linear gradient: {max_err:.2e}")


if __name__ == "__main__":
    main()
