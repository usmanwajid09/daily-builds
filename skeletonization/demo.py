"""Demo: thin a thick plus sign and a thick bar to their skeletons.

Builds a 31x31 binary image with a thick plus sign (arms 5 pixels
wide) and a thick horizontal bar, runs Zhang-Suen thinning, and
prints the skeleton pixel counts plus the endpoint and branch-point
counts. A plus sign's skeleton is a cross: 4 endpoints, and a small
junction cluster that counts as 5 branch points under 8-connectivity
(the centre plus its four diagonal neighbours). A bar's skeleton is
a single line: 2 endpoints, 0 branch points.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from skeletonization.skeletonization import (
    count_branch_points,
    count_endpoints,
    zhang_suen_thin,
)


def make_plus(size=31, arm=5):
    """Return a binary image holding a thick plus sign."""
    image = np.zeros((size, size), dtype=bool)
    mid = size // 2
    half = arm // 2
    image[mid - half:mid + half + 1, 3:size - 3] = True   # horizontal arm
    image[3:size - 3, mid - half:mid + half + 1] = True   # vertical arm
    return image


def make_bar(rows=31, cols=31, thickness=7):
    """Return a binary image holding a thick horizontal bar."""
    image = np.zeros((rows, cols), dtype=bool)
    image[12:12 + thickness, 5:26] = True
    return image


def _report(name, binary):
    skeleton, iterations = zhang_suen_thin(binary)
    print(f"{name}: {int(binary.sum())} foreground pixels "
          f"-> {int(skeleton.sum())} skeleton pixels "
          f"({iterations} iterations)")
    print(f"  endpoints={count_endpoints(skeleton)}, "
          f"branch points={count_branch_points(skeleton)}")


def main():
    _report("plus sign", make_plus())
    _report("bar", make_bar())


if __name__ == "__main__":
    main()
