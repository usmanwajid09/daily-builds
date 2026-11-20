"""Demo: detect lane lines on a synthetic road image.

Generates a 320x240 road with two lane lines converging to a vanishing
point (plus a dashed centre line and Gaussian noise), runs the
Sobel + Hough detector, and prints the detected endpoints and their
error against the true lane geometry.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from road_lane_detector.model import detect_lanes, make_road_image

SEED = 5
NOISE = 5.0


def _endpoint_error(detected, truth):
    return max(abs(detected[0][0] - truth[0][0]),
               abs(detected[1][0] - truth[1][0]))


def main():
    image, truth = make_road_image(noise=NOISE, seed=SEED)
    result = detect_lanes(image)
    print(f"image: {image.shape[1]}x{image.shape[0]}, noise sigma: {NOISE}")
    for side in ("left", "right"):
        detected = result[side]
        votes = result["votes"][side]
        if detected is None:
            print(f"  {side}: NOT DETECTED (votes {votes})")
            continue
        err = _endpoint_error(detected, truth[side])
        print(f"  {side}: bottom ({detected[0][0]:.1f}, {detected[0][1]:.0f}) "
              f"-> horizon ({detected[1][0]:.1f}, {detected[1][1]:.0f}), "
              f"votes {votes}, max endpoint error {err:.1f} px")
    return result


if __name__ == "__main__":
    main()
