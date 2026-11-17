"""Demo: running-average background subtraction on synthetic video.

Builds a 40x60 background (value 50 plus small seeded noise) and a
10-frame video: the first 5 frames are static, then a bright 10x10
square (value 200) moves 3 pixels to the right each frame.  Runs
:class:`background_subtraction.BackgroundSubtractor` and prints the
foreground pixel count per frame.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from background_subtraction.background_subtraction import BackgroundSubtractor


def make_video(n_frames=10, height=40, width=60):
    """Return a list of frames: static background, then a moving square."""
    rng = np.random.default_rng(0)
    background = 50.0 + rng.normal(0, 2.0, size=(height, width))
    frames = []
    for i in range(n_frames):
        frame = background.copy()
        if i >= 5:
            x0 = 5 + 3 * (i - 5)
            frame[15:25, x0:x0 + 10] = 200.0
        frames.append(frame)
    return frames


def main():
    frames = make_video()
    subtractor = BackgroundSubtractor(learning_rate=0.1, threshold=25.0)
    print(f"video: {len(frames)} frames of {frames[0].shape}")
    for i, frame in enumerate(frames):
        mask = subtractor.apply(frame)
        print(f"  frame {i}: {int(mask.sum())} foreground pixels")
    bg = subtractor.get_background()
    print(f"background model mean: {bg.mean():.2f} (true background ~50)")


if __name__ == "__main__":
    main()
