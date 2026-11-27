"""Demo: mean-shift segmentation of a synthetic RGB scene.

Run from the repo root:  python mean_shift_segmentation/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mean_shift_segmentation import (
    mean_shift_segment,
    segment_count,
    segment_image,
)


def build_scene(height=32, width=40):
    """Synthetic scene: sky, a red rectangle, a green disc, a blue block."""
    scene = np.zeros((height, width, 3), dtype=np.uint8)
    scene[:, :] = (135, 206, 235)          # sky
    scene[6:16, 4:16] = (220, 30, 30)     # red rectangle
    yy, xx = np.mgrid[0:height, 0:width]
    disc = (yy - 24) ** 2 + (xx - 28) ** 2 <= 6 ** 2
    scene[disc] = (30, 180, 60)           # green disc
    scene[22:30, 6:14] = (40, 60, 200)    # blue block
    rng = np.random.default_rng(0)
    noise = rng.integers(-5, 6, size=scene.shape)
    return np.clip(scene.astype(int) + noise, 0, 255).astype(np.uint8)


def main():
    scene = build_scene()
    labels = mean_shift_segment(scene, bandwidth=0.25, spatial_weight=0.5)
    flat = segment_image(scene, bandwidth=0.25, spatial_weight=0.5)

    print(f"Scene shape: {scene.shape}, dtype: {scene.dtype}")
    print(f"Segments found: {segment_count(labels)}")
    sizes = np.bincount(labels.ravel())
    print(f"Segment sizes (pixels): {sorted(sizes.tolist(), reverse=True)}")
    print(f"Segmented image shape: {flat.shape}, dtype: {flat.dtype}")
    print(f"Distinct colours in segmented image: "
          f"{len(np.unique(flat.reshape(-1, 3), axis=0))}")


if __name__ == "__main__":
    main()
