"""Demo: motion detector on a synthetic 64x64 video.

Builds 5 static background frames (value 0.2), then 5 frames where a
bright 10x10 square (value 1.0) moves +4 px to the right each frame.
Runs :class:`motion_detector.MotionDetector` and prints, per frame,
the number of bounding boxes, the motion pixel count and (for moving
frames) the detected box and its centre versus the square's true
centre.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from motion_detector.motion_detector import MotionDetector

HEIGHT, WIDTH = 64, 64
BG_VALUE = 0.2
SQUARE_VALUE = 1.0
SQUARE_SIZE = 10
SQUARE_Y = 27
SQUARE_X_START = 8
STEP = 4
N_STATIC = 5
N_MOVING = 5


def make_frames():
    """Return (frames, true_centres): static frames, then a moving square.

    true_centres[k] is the (y, x) centre of the square in moving
    frame k, or None for static frames.
    """
    background = np.full((HEIGHT, WIDTH), BG_VALUE)
    frames, centres = [], []
    for _ in range(N_STATIC):
        frames.append(background.copy())
        centres.append(None)
    for k in range(N_MOVING):
        frame = background.copy()
        x0 = SQUARE_X_START + STEP * k
        frame[SQUARE_Y:SQUARE_Y + SQUARE_SIZE,
              x0:x0 + SQUARE_SIZE] = SQUARE_VALUE
        frames.append(frame)
        centres.append((SQUARE_Y + (SQUARE_SIZE - 1) / 2,
                        x0 + (SQUARE_SIZE - 1) / 2))
    return frames, centres


def main():
    frames, true_centres = make_frames()
    detector = MotionDetector(threshold=0.1, learning_rate=0.01, min_area=20)
    # learning_rate is deliberately small: the square overlaps its
    # previous position by 6 columns each step, and a faster update
    # would leave a faint "ghost" of the old position in the
    # background that exceeds the threshold.

    static_boxes = 0
    static_pixels = 0
    moving_boxes = []
    print("frame | boxes | motion_pixels | box (y_min,x_min,y_max,x_max) | centre (y,x) vs true")
    print("------+-------+---------------+--------------------------------+----------------------")
    for i, frame in enumerate(frames):
        result = detector.detect(frame)
        boxes = result["bounding_boxes"]
        pixels = result["motion_pixels"]
        if i < N_STATIC:
            static_boxes += len(boxes)
            static_pixels += pixels
            print(f"{i:5d} | {len(boxes):5d} | {pixels:13d} | static frame")
        else:
            moving_boxes.append(len(boxes))
            box = boxes[0] if boxes else None
            if box is not None:
                centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                print(f"{i:5d} | {len(boxes):5d} | {pixels:13d} | {box} | "
                      f"({centre[0]:.1f}, {centre[1]:.1f}) vs {true_centres[i]}")
            else:
                print(f"{i:5d} | {len(boxes):5d} | {pixels:13d} | NO BOX DETECTED")

    print()
    print(f"Static frames: {static_boxes} boxes, {static_pixels} motion pixels "
          f"total (expected 0 / 0 after warm-up).")
    print(f"Moving frames: {moving_boxes} boxes per frame "
          f"(expected [1, 1, 1, 1, 1]).")
    print(f"Square area: {SQUARE_SIZE * SQUARE_SIZE} px; "
          f"motion pixels on moving frames should be ~100 each.")


if __name__ == "__main__":
    main()
