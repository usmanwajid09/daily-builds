"""Demo: Lucas-Kanade optical flow on a synthetic moving blob.

Builds a 60x80 frame with a Gaussian blob centred at (40, 30) and a
second frame with the same blob shifted by (dx=3, dy=2), then estimates
the flow at the blob centre and on a coarse grid with
:func:`optical_flow.dense_flow`.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from optical_flow.optical_flow import dense_flow, lucas_kanade


def make_frames(height=60, width=80, dx=3, dy=2):
    """Return two frames with a Gaussian blob shifted by (dx, dy)."""
    yy, xx = np.mgrid[0:height, 0:width]
    sigma = 8.0
    prev = 200.0 * np.exp(-((xx - 40.0) ** 2 + (yy - 30.0) ** 2) / (2 * sigma ** 2))
    curr = 200.0 * np.exp(-((xx - (40.0 + dx)) ** 2 + (yy - (30.0 + dy)) ** 2) / (2 * sigma ** 2))
    return prev, curr


def main():
    prev, curr = make_frames()
    flow = lucas_kanade(prev, curr, [(40, 30)], window_size=15)
    print(f"true shift: (dx=3, dy=2)")
    print(f"flow at blob centre (40, 30): u={flow[0, 0]:.3f}, v={flow[0, 1]:.3f}")
    u, v, xs, ys = dense_flow(prev, curr, window_size=15, step=10)
    magnitude = np.hypot(u, v)
    print(f"dense flow grid: {u.shape[0]}x{u.shape[1]} points")
    print(f"mean flow magnitude over grid: {magnitude.mean():.3f}")
    print(f"max flow magnitude over grid: {magnitude.max():.3f}")
    cy, cx = len(ys) // 2, len(xs) // 2
    print(f"flow at grid centre ({xs[cx]}, {ys[cy]}): u={u[cy, cx]:.3f}, v={v[cy, cx]:.3f}")


if __name__ == "__main__":
    main()
