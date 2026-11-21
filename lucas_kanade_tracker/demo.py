"""Demo: Lucas-Kanade tracking on a moving textured disc.

A disc filled with fixed random texture moves by a known amount
(dy=+2, dx=+3) between two synthetic frames. Points inside the disc
are tracked and the estimated displacement is compared with the
truth; points in the flat background are correctly reported lost.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lucas_kanade_tracker import textured_disc_frame, track_points


def main():
    true_shift = (2.0, 3.0)  # (dy, dx)
    prev = textured_disc_frame(center=(32, 32), radius=12, seed=0)
    curr = textured_disc_frame(center=(34, 35), radius=12, seed=0)
    rng = np.random.default_rng(5)
    angles = rng.uniform(0, 2 * np.pi, size=8)
    radii = rng.uniform(0, 4, size=8)
    points = np.column_stack([
        32 + radii * np.sin(angles),
        32 + radii * np.cos(angles),
    ])
    flat_points = np.array([[5.0, 5.0], [60.0, 60.0], [5.0, 58.0]])
    all_points = np.vstack([points, flat_points])
    new_points, status, errors = track_points(prev, curr, all_points)
    displacement = new_points[:8] - points
    mean_disp = displacement.mean(axis=0)
    err = np.hypot(*(mean_disp - np.array(true_shift)))
    print(f"true shift: dy={true_shift[0]:.0f}, dx={true_shift[1]:.0f}")
    print(f"tracked points: {int(status[:8].sum())}/8, "
          f"mean displacement: dy={mean_disp[0]:.2f}, dx={mean_disp[1]:.2f}")
    print(f"mean displacement error: {err:.3f} px")
    print(f"mean residual in window: {np.nanmean(errors[:8]):.4f}")
    print(f"flat-background points lost as expected: "
          f"{int((status[8:] == 0).sum())}/3")
    return new_points, status, errors


if __name__ == "__main__":
    main()
