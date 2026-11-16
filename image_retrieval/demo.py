"""Demo: content-based image retrieval by colour histogram.

Builds an index of 20 synthetic images in 5 colour themes (sunset,
forest, ocean, desert, night — 4 variants each), then queries the index
with a fresh "ocean" image and prints the top-5 ranked results with
their histogram-correlation scores. A second check queries with an exact
copy of an indexed image, which must score 1.0 and rank first.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python image_retrieval/demo.py`):
# make the repo root importable so `image_retrieval` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_retrieval.retrieval import (
    ImageIndex,
    make_blocks,
    make_gradient,
    make_solid,
)

THEMES = {
    "sunset": [(255, 120, 40), (255, 180, 90), (200, 60, 30), (255, 220, 150)],
    "forest": [(20, 120, 40), (40, 160, 60), (10, 90, 30), (120, 180, 80)],
    "ocean": [(20, 90, 180), (30, 130, 200), (10, 60, 140), (90, 170, 220)],
    "desert": [(220, 180, 100), (200, 150, 70), (240, 210, 140), (180, 130, 60)],
    "night": [(20, 20, 60), (40, 40, 90), (10, 10, 40), (70, 70, 120)],
}


def _theme_image(theme, rng):
    colors = THEMES[theme]
    kind = rng.integers(3)
    if kind == 0:
        return make_gradient(colors[0], colors[1])
    if kind == 1:
        return make_blocks(colors, rng=rng)
    return make_solid(colors[int(rng.integers(len(colors)))])


def main():
    rng = np.random.default_rng(21)
    index = ImageIndex()
    indexed = {}
    for theme in THEMES:
        for variant in range(4):
            image = _theme_image(theme, rng)
            name = f"{theme}-{variant}"
            index.add(name, image)
            indexed[name] = image

    print(f"index: {len(index)} images across {len(THEMES)} themes")

    query = _theme_image("ocean", rng)
    results = index.query(query, top_k=5)
    print("query: fresh 'ocean' image — top 5 results:")
    for rank, (name, score) in enumerate(results, start=1):
        theme = name.rsplit("-", 1)[0]
        print(f"  {rank}. {name:12s} score={score:.4f} "
              f"{'(same theme)' if theme == 'ocean' else ''}")
    top_theme = results[0][0].rsplit("-", 1)[0]
    print(f"top-1 theme correct: {top_theme == 'ocean'}")

    exact = indexed["forest-2"]
    self_results = index.query(exact, top_k=1)
    print(f"self-query: '{self_results[0][0]}' score={self_results[0][1]:.4f} "
          f"(expected forest-2 at 1.0000)")
    return results


if __name__ == "__main__":
    main()
