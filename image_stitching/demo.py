"""Demo: panorama stitching from scratch on synthetic images.

Builds two 64x64 grayscale images cut from one larger synthetic
scene (a gradient plus rectangles, a circle and seeded noise), where
img2 is the same scene shifted by (dy=+5, dx=+20) relative to img1.
Recovers the shift with :func:`find_best_shift`, stitches with
:func:`stitch_images`, and prints verifiable numbers: the found shift
against the true shift, the NCC score, the panorama shape, and checks
that the non-overlap region is copied exactly and the seam is smooth.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from image_stitching.image_stitching import find_best_shift, stitch_images


def make_scene(height=69, width=84, seed=42):
    """Synthetic scene: gradient + shapes + noise (non-periodic)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    scene = 0.6 * xx + 0.4 * yy  # diagonal gradient
    scene[10:25, 10:30] += 120.0  # bright rectangle
    scene[40:55, 50:70] += 90.0  # second rectangle
    circle = (yy - 32) ** 2 + (xx - 60) ** 2 < 8 ** 2
    scene[circle] += 150.0  # bright disc
    scene += rng.random((height, width)) * 30.0  # texture / noise
    return scene


def main():
    scene = make_scene()
    img1 = scene[0:64, 0:64]
    img2 = scene[5:69, 20:84]
    true_shift = (5, 20)  # (dy, dx) of img2 relative to img1

    found_shift, score = find_best_shift(img1, img2)
    pano = stitch_images(img1, img2, shift=found_shift)

    # Non-overlap check: panorama must equal img1 exactly where only
    # img1 covers the canvas (rows 0-4 and columns 0-19 of img1).
    img1_only = pano[0:5, 0:64]  # rows above img2, full img1 width
    non_overlap_diff = np.max(np.abs(img1_only - img1[0:5, :]))

    # Seam check: both source images come from the same scene, so the
    # panorama should reproduce the scene everywhere it is covered
    # (blend of equal values is exact). Note the bottom-left corner
    # (rows 64-68, cols 0-19) is covered by neither image, so it is
    # excluded -- it stays 0 on the canvas.
    covered = np.zeros((69, 84), dtype=bool)
    covered[0:64, 0:64] = True  # img1
    covered[5:69, 20:84] = True  # img2
    scene_diff = np.max(np.abs(pano[covered] - scene[covered]))
    seam_step = np.mean(np.abs(pano[:, 20] - pano[:, 19]))
    typical_step = np.mean(np.abs(np.diff(scene, axis=1)))

    print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}")
    print(f"true shift (dy, dx): {true_shift}")
    print(f"found shift (dy, dx): {found_shift}")
    print(f"shift recovered exactly: {found_shift == true_shift}")
    print(f"NCC score at best shift: {score:.6f}")
    print(f"panorama shape: {pano.shape}")
    print(f"max |panorama - img1| in img1-only region: {non_overlap_diff:.6f}")
    print(f"max |panorama - scene| over covered region: {scene_diff:.6f}")
    print(f"mean |step| across seam column: {seam_step:.4f}")
    print(f"mean |step| of scene itself: {typical_step:.4f}")
    return pano


if __name__ == "__main__":
    main()
