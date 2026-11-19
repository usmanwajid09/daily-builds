"""Demo: stereo disparity on a synthetic stereo pair.

Builds a synthetic pair with a near square (disparity 8) and a far
square (disparity 3) over a textured background, runs SAD block
matching, and reports the accuracy against ground truth plus the
median disparity in each square — the near square must come out
larger than the far one, which is the depth-ordering check.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from stereo_disparity.model import (
    block_match,
    depth_order_ok,
    disparity_accuracy,
    make_synthetic_pair,
)


def main():
    left, right, truth = make_synthetic_pair()
    disparity = block_match(left, right, max_disparity=12, window_radius=2)
    print(f"Image size: {left.shape[0]}x{left.shape[1]}")
    valid = disparity >= 0
    print(f"Valid pixels: {int(valid.sum())}/{disparity.size}")
    print(f"Disparity accuracy vs ground truth: "
          f"{disparity_accuracy(disparity, truth):.3f}")
    near = disparity[37:53, 47:63]
    far = disparity[12:28, 12:28]
    print(f"Median disparity — near square: {np.median(near[near >= 0]):.0f} "
          f"(true 8), far square: {np.median(far[far >= 0]):.0f} (true 3)")
    ok = depth_order_ok(disparity, (slice(37, 53), slice(47, 63)),
                        (slice(12, 28), slice(12, 28)))
    print(f"Depth ordering check (near > far): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
