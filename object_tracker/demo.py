"""Demo: track a moving disc across synthetic frames with NCC.

Generates 20 frames of a bright disc moving at (3, 2) pixels per
frame, initialises the tracker on the first frame, tracks the rest,
and prints the mean drift (px) against the true path, the final
position error, and the worst NCC score.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from object_tracker.model import (
    ObjectTracker,
    bbox_centre,
    make_sequence,
    mean_drift,
)

N_FRAMES = 20
START = (30.0, 30.0)
VELOCITY = (3.0, 2.0)
RADIUS = 10


def main():
    frames, true_centres = make_sequence(
        n_frames=N_FRAMES, start=START, velocity=VELOCITY, radius=RADIUS,
        noise=2.0, seed=3)
    first = true_centres[0]
    bbox = (int(first[0]) - RADIUS, int(first[1]) - RADIUS,
            2 * RADIUS + 1, 2 * RADIUS + 1)
    tracker = ObjectTracker(search_radius=15, min_score=0.5).init(frames[0], bbox)
    results = tracker.track_sequence(frames[1:])
    tracked = [bbox] + [b for b, _ in results]
    scores = [s for _, s in results]
    drift = mean_drift(tracked, true_centres)
    final_err = float(np.linalg.norm(
        bbox_centre(tracked[-1]) - true_centres[-1]))
    print(f"frames: {N_FRAMES}, velocity: {VELOCITY} px/frame, "
          f"template: {2 * RADIUS + 1}x{2 * RADIUS + 1}")
    print(f"mean drift vs true path: {drift:.3f} px")
    print(f"final position error: {final_err:.3f} px")
    print(f"worst NCC score: {min(scores):.4f} "
          f"(lost frames: {sum(s < 0.5 for s in scores)})")
    return drift


if __name__ == "__main__":
    main()
