"""Demo: train the detector on synthetic circles, then find them in a scene."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from sliding_window_detector.detector import (
    SlidingWindowDetector,
    make_circle_window,
    make_negative_window,
    make_scene,
)


def main():
    rng = np.random.default_rng(0)
    # Positives: circles with small centre jitter, so the detector tolerates
    # circles that are not perfectly centred in the window.
    positives = []
    for _ in range(60):
        cy = 8 + int(rng.integers(-2, 3))
        cx = 8 + int(rng.integers(-2, 3))
        positives.append(make_circle_window(16, centre=(cy, cx),
                                            radius=int(rng.integers(4, 6)), rng=rng))
    negatives = []
    for kind in ("blank", "noise", "square"):
        for _ in range(20):
            negatives.append(make_negative_window(16, kind=kind, rng=rng))

    detector = SlidingWindowDetector(window_size=16, stride=4, threshold=0.7)
    detector.fit(positives, negatives)
    print(f"Trained on {len(positives)} positive and {len(negatives)} negative windows.")
    print(f"HOG feature length per window: "
          f"{len(detector._features(positives[0]))}")

    scene = make_scene(size=48, circles=((16, 16), (34, 32)), radius=5, rng=rng)
    detections = detector.detect(scene)
    print(f"Scene: 48x48 with circles centred at (16, 16) and (34, 32).")
    print(f"Detections after NMS: {len(detections)}")
    for det in detections:
        print(f"  box x={det['x']} y={det['y']} {det['w']}x{det['h']} "
              f"score={det['score']:.3f}")

    blank = np.zeros((48, 48))
    print(f"Detections on a blank scene: {len(detector.detect(blank))} (expected 0)")


if __name__ == "__main__":
    main()
