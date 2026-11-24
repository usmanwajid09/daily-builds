"""Demo: CamShift tracking of a moving coloured disc.

Builds 12 synthetic 80x100 frames: a red disc moves diagonally across
a green background. CamShift is initialised from a small box fully
inside the disc in frame 0 and must follow the disc; the demo prints
the tracked centre and its error against the true centre.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camshift_tracker.camshift import build_histogram, track_sequence


def make_frames(n_frames=12, height=80, width=100):
    """Return (frames, true_centres): red disc moving down-right."""
    frames, centres = [], []
    for i in range(n_frames):
        cx, cy = 25 + 4 * i, 20 + 3 * i
        frame = np.zeros((height, width, 3), dtype=np.float64)
        frame[:, :, 1] = 120.0  # green background
        rows, cols = np.mgrid[0:height, 0:width]
        disc = (rows - cy) ** 2 + (cols - cx) ** 2 <= 10 ** 2
        frame[disc] = [220.0, 30.0, 30.0]  # red disc
        frames.append(frame)
        centres.append((cx, cy))
    return frames, centres


def main():
    frames, centres = make_frames()
    init_window = (20, 15, 10, 10)  # small box fully inside the disc
    hist = build_histogram(frames[0], init_window, bins=16)
    windows = track_sequence(frames, init_window, hist)
    print(f"frames: {len(frames)}, initial window: {init_window}")
    for i, (window, (tx, ty)) in enumerate(zip(windows, centres)):
        cx, cy = window[0] + window[2] / 2.0, window[1] + window[3] / 2.0
        err = float(np.hypot(cx - tx, cy - ty))
        print(f"frame {i:2d}: window={window} centre=({cx:.1f}, {cy:.1f}) "
              f"true=({tx}, {ty}) error={err:.2f}px")
    final = windows[-1]
    fcx, fcy = final[0] + final[2] / 2.0, final[1] + final[3] / 2.0
    tx, ty = centres[-1]
    print(f"final tracking error: {np.hypot(fcx - tx, fcy - ty):.2f}px")
    return windows


if __name__ == "__main__":
    main()
