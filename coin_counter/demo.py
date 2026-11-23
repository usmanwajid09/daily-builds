"""Demo: count coins in a synthetic image.

Generates a 200x200 scene with a known set of coins (3x radius 10,
2x radius 14, 1x radius 18), runs :func:`coin_counter.count_coins`
on it, and prints the detected count, the per-denomination
breakdown and the total value next to the ground truth.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python coin_counter/demo.py`):
# make the repo root importable so `coin_counter` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coin_counter.coins import RADIUS_TO_VALUE, count_coins, make_coins_image

# Ground truth: (x, y, radius) -- 3x radius 10 (value 1),
# 2x radius 14 (value 5), 1x radius 18 (value 10).
GROUND_TRUTH = [
    (40, 40, 10),
    (100, 35, 10),
    (160, 45, 10),
    (50, 130, 14),
    (130, 120, 14),
    (165, 155, 18),
]
SEED = 7


def main():
    image = make_coins_image(GROUND_TRUTH, size=200, seed=SEED)
    coins, summary = count_coins(image)

    expected_total = sum(RADIUS_TO_VALUE[r] for _, _, r in GROUND_TRUTH)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, "
          f"seed={SEED}")
    print(f"ground truth: {len(GROUND_TRUTH)} coins, "
          f"total value {expected_total}")
    print(f"detected:     {summary['count']} coins, "
          f"total value {summary['total_value']}")
    for coin in coins:
        print(f"  coin at ({coin['x']}, {coin['y']}), "
              f"radius {coin['radius']:.1f}, "
              f"denomination {coin['denomination']}")
    print("per-denomination breakdown "
          f"(detected): {summary['by_denomination']}")
    expected_breakdown = {}
    for _, _, r in GROUND_TRUTH:
        value = RADIUS_TO_VALUE[r]
        expected_breakdown[value] = expected_breakdown.get(value, 0) + 1
    print(f"per-denomination breakdown (truth): {expected_breakdown}")
    match = (
        summary["count"] == len(GROUND_TRUTH)
        and summary["total_value"] == expected_total
        and summary["by_denomination"] == expected_breakdown
    )
    print(f"exact match with ground truth: {match}")
    return match


if __name__ == "__main__":
    main()
