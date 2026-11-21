"""Demo: colour constancy on a synthetic image with a known cast.

Builds a neutral synthetic scene (a colour gradient plus shapes),
applies a strong warm cast (red boosted, blue cut), then corrects it
with gray-world and white-patch and prints the channel means before
and after, plus the estimated illuminant.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from color_constancy.color_constancy import (
    estimate_illuminant,
    white_balance,
)


def make_scene(seed=0):
    """A neutral synthetic RGB scene: gradient sky, green field, grey road."""
    rng = np.random.default_rng(seed)
    height, width = 120, 160
    y, x = np.mgrid[0:height, 0:width]
    scene = np.zeros((height, width, 3), dtype=np.float64)
    scene[:, :, 0] = 120 + 60 * np.sin(x / 25.0)          # red
    scene[:, :, 1] = 130 + 50 * np.cos(y / 20.0)          # green
    scene[:, :, 2] = 140 + 40 * np.sin((x + y) / 30.0)    # blue
    scene[80:, :, 0] = 90                                # grey road
    scene[80:, :, 1] = 95
    scene[80:, :, 2] = 100
    scene += rng.normal(0.0, 5.0, size=scene.shape)
    return np.clip(scene, 0, 255).astype(np.uint8)


def main():
    scene = make_scene()
    cast = np.array([1.45, 1.0, 0.55])  # warm cast: red up, blue down
    tinted = np.clip(
        scene.astype(np.float64) * cast, 0, 255
    ).astype(np.uint8)

    def means(img):
        return img.reshape(-1, 3).mean(axis=0)

    print(f"original scene means (R,G,B): {np.round(means(scene), 1)}")
    print(f"tinted image means (R,G,B):   {np.round(means(tinted), 1)}")
    print(f"true cast applied:            {cast}")
    for method in ("gray_world", "white_patch", "shades_of_gray"):
        corrected = white_balance(tinted, method=method)
        illum = estimate_illuminant(tinted, method=method)
        spread = means(corrected).max() - means(corrected).min()
        err = np.abs(corrected.astype(np.float64) - scene).mean()
        print(
            f"{method:15s} corrected means: {np.round(means(corrected), 1)} "
            f"(spread {spread:5.2f}), estimated illuminant "
            f"{np.round(illum, 3)}, mean |error| vs scene: {err:.2f}"
        )
    return tinted


if __name__ == "__main__":
    main()
