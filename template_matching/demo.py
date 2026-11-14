"""Demo: template matching on a synthetic image.

Builds a 60x80 random-noise image, cuts a 12x10 patch out of it and
pastes the exact patch back in at a different, known location, then
runs :func:`template_matching.match.match_template` and prints the
found location against the true one plus the score. An exact copy
should score 1.0.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python template_matching/demo.py`):
# make the repo root importable so `template_matching` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template_matching.match import match_template, match_template_map

TRUE_ROW, TRUE_COL = 37, 52
TEMPLATE_SHAPE = (12, 10)


def make_test_image(seed=0):
    """Return ``(image, template)`` with the template pasted at the true spot."""
    rng = np.random.default_rng(seed)
    image = rng.random((60, 80)) * 255.0
    # Cut the template from one spot, scrub that spot with fresh noise
    # (so only one exact copy exists), then paste it at the true spot.
    template = image[5:5 + TEMPLATE_SHAPE[0], 20:20 + TEMPLATE_SHAPE[1]].copy()
    image[5:5 + TEMPLATE_SHAPE[0], 20:20 + TEMPLATE_SHAPE[1]] = (
        rng.random(TEMPLATE_SHAPE) * 255.0
    )
    image[
        TRUE_ROW:TRUE_ROW + TEMPLATE_SHAPE[0],
        TRUE_COL:TRUE_COL + TEMPLATE_SHAPE[1],
    ] = template
    return image, template


def main():
    image, template = make_test_image()
    row, col, score = match_template(image, template)
    score_map = match_template_map(image, template)
    print(f"image: {image.shape[0]}x{image.shape[1]} grayscale, "
          f"template: {template.shape[0]}x{template.shape[1]}")
    print(f"true location:    (row={TRUE_ROW}, col={TRUE_COL})")
    print(f"found location:   (row={row}, col={col})")
    print(f"best NCC score:   {score:.6f}")
    print(f"score map shape:  {score_map.shape}, "
          f"range [{score_map.min():.3f}, {score_map.max():.3f}]")
    print("match correct:", (row, col) == (TRUE_ROW, TRUE_COL))
    return row, col, score


if __name__ == "__main__":
    main()
