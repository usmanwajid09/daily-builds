"""Demo: cartoonize a synthetic RGB scene and print the colour counts.

Run from the repo root:  python cartoonizer/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cartoonizer import cartoonize, edge_mask


def build_scene(height=60, width=80):
    """Synthetic RGB scene: sky, a red rectangle, a green disc, a blue block."""
    scene = np.zeros((height, width, 3), dtype=np.uint8)
    scene[:, :] = (135, 206, 235)          # sky blue background
    scene[10:35, 8:30] = (220, 30, 30)    # red rectangle (house)
    yy, xx = np.mgrid[0:height, 0:width]
    disc = (yy - 45) ** 2 + (xx - 60) ** 2 <= 10 ** 2
    scene[disc] = (30, 180, 60)           # green disc (tree)
    scene[40:55, 45:75] = (40, 60, 200)   # blue triangle-ish block (pond)
    # Add a little noise so smoothing has something to do.
    rng = np.random.default_rng(0)
    noise = rng.integers(-6, 7, size=scene.shape)
    return np.clip(scene.astype(int) + noise, 0, 255).astype(np.uint8)


def count_colours(image):
    return len(np.unique(image.reshape(-1, 3), axis=0))


def main():
    scene = build_scene()
    cartoon = cartoonize(scene, n_colors=8, edge_threshold=30.0)
    edges = edge_mask(scene, threshold=30.0)

    print(f"Scene shape: {scene.shape}, dtype: {scene.dtype}")
    print(f"Distinct colours before cartoonize: {count_colours(scene)}")
    print(f"Distinct colours after cartoonize:  {count_colours(cartoon)}")
    print(f"Edge pixels (edge_mask, threshold=30): {int(edges.sum())} "
          f"of {edges.size}")
    print(f"Cartoon output shape: {cartoon.shape}, dtype: {cartoon.dtype}")


if __name__ == "__main__":
    main()
