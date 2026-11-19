"""Demo: convex hull of a fixed set of random points.

Generates 30 random points in a 10x10 square (seed 42), computes
the hull with both the Graham scan and the Jarvis march, and prints
the hull vertices, the hull area, and whether a chosen interior and
exterior point lie inside the hull.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from convex_hull.convex_hull import (
    graham_scan,
    jarvis_march,
    point_in_hull,
    polygon_area,
)


def make_points(n=30, seed=42):
    """Return ``n`` random points in a 10x10 square."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 10, size=(n, 2))


def main():
    points = make_points()
    graham = graham_scan(points)
    jarvis = jarvis_march(points)
    print(f"{len(points)} random points in a 10x10 square (seed 42)")
    print(f"Graham scan hull: {len(graham)} vertices")
    print(f"  vertices: {np.round(graham, 2).tolist()}")
    print(f"Jarvis march hull: {len(jarvis)} vertices, "
          f"same set: {set(map(tuple, jarvis)) == set(map(tuple, graham))}")
    print(f"hull area: {polygon_area(graham):.2f} (square is 100.00)")
    centre = points.mean(axis=0)
    print(f"centroid {np.round(centre, 2)} inside hull: "
          f"{point_in_hull(centre, graham)}")
    print(f"point (20, 20) inside hull: {point_in_hull((20, 20), graham)}")


if __name__ == "__main__":
    main()
