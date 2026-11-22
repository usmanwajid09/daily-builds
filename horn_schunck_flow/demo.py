"""Demo: Horn-Schunck optical flow on a synthetic translated blob.

Builds a 64x64 image with a bright Gaussian blob on a dark background,
shifts the blob by a known amount (dx=+3, dy=+2) to make a second
frame, runs :class:`HornSchunckFlow`, and prints the recovered mean
flow inside the blob versus the true shift, plus the (near-zero) flow
in the static background.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from horn_schunck_flow.horn_schunck_flow import (
    HornSchunckFlow,
    flow_magnitude,
    mean_flow,
)


def make_blob(height=64, width=64, cy=32.0, cx=32.0, sigma=5.0, amplitude=255.0):
    """Bright Gaussian blob on a dark background."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    return amplitude * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))


def main():
    dx_true, dy_true = 3.0, 2.0
    img1 = make_blob()
    img2 = make_blob(cy=32.0 + dy_true, cx=32.0 + dx_true)

    flow = HornSchunckFlow(alpha=1.0, n_iterations=100)
    u, v = flow.compute_flow(img1, img2)

    # Object region: pixels that are bright in frame 1 (the blob core).
    obj_mask = img1 > 50.0
    # Static background: the four 8x8 corner blocks, far from the blob
    # (the Gaussian tail is ~0 there and the flow has not diffused in).
    bg_mask = np.zeros_like(img1, dtype=bool)
    bg_mask[:8, :8] = True
    bg_mask[:8, -8:] = True
    bg_mask[-8:, :8] = True
    bg_mask[-8:, -8:] = True
    mean_u_obj = float(np.mean(u[obj_mask]))
    mean_v_obj = float(np.mean(v[obj_mask]))
    mag = flow_magnitude(u, v)
    mean_mag_bg = float(np.mean(mag[bg_mask]))
    mean_u_all, mean_v_all = mean_flow(u, v)

    print(f"image: {img1.shape[0]}x{img1.shape[1]} grayscale, blob shifted by (dx, dy) = ({dx_true:.0f}, {dy_true:.0f})")
    print(f"iterations run: {flow.iterations_run_}")
    print(f"true shift: dx = {dx_true:.2f} px, dy = {dy_true:.2f} px")
    print(f"recovered mean flow in object region: u = {mean_u_obj:.3f} px, v = {mean_v_obj:.3f} px")
    print(f"recovery error: du = {abs(mean_u_obj - dx_true):.3f} px, dv = {abs(mean_v_obj - dy_true):.3f} px")
    print(f"mean flow magnitude in static background: {mean_mag_bg:.4f} px")
    print(f"overall mean flow: u = {mean_u_all:.3f} px, v = {mean_v_all:.3f} px")
    print(f"overall mean flow magnitude: {float(np.mean(mag)):.3f} px")
    return u, v


if __name__ == "__main__":
    main()
