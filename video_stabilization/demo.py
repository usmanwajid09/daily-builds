"""Demo: stabilise a jittery synthetic video.

Builds 21 frames of a 48x64 scene (a bright square on a textured
background) that drifts smoothly while shaking with random jitter,
then stabilises the sequence and compares the frame-to-frame jitter
before and after.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from video_stabilization.stabilization import (
    estimate_translation,
    stabilize_frames,
)


def make_test_frames(n_frames=21, height=48, width=64, seed=0):
    """Return jittery frames of a drifting scene with a bright square."""
    rng = np.random.default_rng(seed)
    base = rng.normal(100.0, 20.0, size=(height, width))
    base[20:30, 28:38] = 230.0
    frames = []
    for i in range(n_frames):
        drift = int(round(0.3 * i))          # smooth deliberate drift
        jitter = int(rng.integers(-2, 3))    # random shake
        total = drift + jitter
        shifted = np.roll(base, shift=total, axis=0)
        shifted = np.roll(shifted, shift=total, axis=1)
        frames.append(shifted)
    return frames


def _jitter_std(frames):
    """Std of frame-to-frame translation after removing the mean drift."""
    steps = np.array([estimate_translation(frames[i - 1], frames[i])
                      for i in range(1, len(frames))], dtype=np.float64)
    return float(steps.std(axis=0).mean()), steps


def main():
    frames = make_test_frames()
    before, _ = _jitter_std(frames)
    stable = stabilize_frames(frames, window=5)
    after, _ = _jitter_std(stable)
    print(f"frames: {len(frames)} of {frames[0].shape[0]}x{frames[0].shape[1]} grayscale")
    print(f"frame-to-frame jitter std before: {before:.2f} px")
    print(f"frame-to-frame jitter std after:  {after:.2f} px")
    print(f"jitter reduced by: {(1 - after / before) * 100:.0f}%")
    return stable


if __name__ == "__main__":
    main()
