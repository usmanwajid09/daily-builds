"""Demo: mean-shift tracking of a moving red square.

A bright red square moves by (dy=+2, dx=+3) per frame over a dark
grey noisy background. The tracker is initialised on the first
frame's bounding box and then follows the square for five frames;
the tracked centre is compared with the true centre each frame.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mean_shift_tracker import MeanShiftTracker, moving_square_frame


def main():
    tracker = MeanShiftTracker()
    center = np.array([30.0, 40.0])
    step = np.array([2.0, 3.0])
    size = 13
    frame = moving_square_frame(center=tuple(center), size=size, seed=0)
    bbox = (center[0] - size // 2, center[1] - size // 2, size, size)
    tracker.init(frame, bbox)
    tracked0 = np.array(tracker.center_)
    err0 = float(np.hypot(*(tracked0 - center)))
    print(f"frame 0: true centre=({center[0]:.0f}, {center[1]:.0f}), "
          f"tracked centre=({tracked0[0]:.2f}, {tracked0[1]:.2f}), "
          f"error={err0:.2f} px")
    max_error = err0
    for i in range(1, 6):
        center = center + step
        frame = moving_square_frame(center=tuple(center), size=size, seed=0)
        tracker.update(frame)
        tracked = np.array(tracker.center_)
        error = float(np.hypot(*(tracked - center)))
        max_error = max(max_error, error)
        print(f"frame {i}: true centre=({center[0]:.0f}, {center[1]:.0f}), "
              f"tracked centre=({tracked[0]:.2f}, {tracked[1]:.2f}), "
              f"error={error:.2f} px")
    print(f"max centre error over 5 frames: {max_error:.2f} px")
    return max_error


if __name__ == "__main__":
    main()
