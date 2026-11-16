"""Demo: FAST corner detection on a synthetic image.

Builds a black image with a white square and a white right triangle (whose
two acute corners are sharp enough for FAST-12), plus a little noise, then
runs the detector with FAST-9 and with the stricter default FAST-12 and
compares both against a uniform, corner-free image.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fast_corner_detector.fast import detect_corners

# True corners of the synthetic shapes (row, col).
SQUARE_CORNERS = [(20, 20), (20, 49), (49, 20), (49, 49)]
TRIANGLE_ACUTE_CORNERS = [(20, 90), (59, 129)]


def make_test_image():
    """Synthetic 100x160 image: white square + white right triangle on black.

    The triangle has vertices (20, 90), (59, 90) and (59, 129), so its two
    acute 45-degree corners are at (20, 90) and (59, 129). A fixed seed
    keeps the added noise (std 3) identical on every run.
    """
    rng = np.random.default_rng(42)
    img = np.zeros((100, 160), dtype=np.float64)
    img[20:50, 20:50] = 255.0  # white square
    for r in range(20, 60):
        for c in range(90, 130):
            if c - 90 <= r - 20:
                img[r, c] = 255.0  # white right triangle
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255)


def make_uniform_image(rows=50, cols=60, value=128.0):
    """A flat image with no corners at all."""
    return np.full((rows, cols), value, dtype=np.float64)


def main():
    image = make_test_image()
    print(f"Image size: {image.shape[0]}x{image.shape[1]}")

    corners9, scores9 = detect_corners(image, threshold=20, n_contiguous=9)
    print(f"\nFAST-9  (n_contiguous=9):  {len(corners9)} corners found")
    for (r, c), s in list(zip(corners9, scores9))[:8]:
        print(f"  corner at (row={r}, col={c}), score={s:.0f}")

    corners12, scores12 = detect_corners(image, threshold=20)  # default N=12
    print(f"\nFAST-12 (n_contiguous=12): {len(corners12)} corners found")
    for (r, c), s in list(zip(corners12, scores12))[:8]:
        print(f"  corner at (row={r}, col={c}), score={s:.0f}")
    print("  (FAST-12 only fires on the triangle's acute corners; a perfect")
    print("   90-degree square corner makes at most an 11-pixel arc.)")

    uniform = make_uniform_image()
    corners_u, _ = detect_corners(uniform, threshold=20, n_contiguous=9)
    print(f"\nUniform {uniform.shape[0]}x{uniform.shape[1]} image: "
          f"{len(corners_u)} corners found (expected 0)")


if __name__ == "__main__":
    main()
