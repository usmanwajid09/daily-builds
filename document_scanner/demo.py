"""Demo: scan a synthetic photographed document (numpy only).

Builds a dark 200x200 image with a bright rotated quadrilateral (the
"document"), finds its four corners, warps it flat, and prints the
recovered corners, the error against the true corners, and the
brightness of the warped output. Deterministic (fixed seed).
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from document_scanner.document_scanner import find_document_corners, scan_document, warp_document

import numpy as np


def make_photo(seed=42):
    """Dark image with a bright rotated quadrilateral; returns (img, corners)."""
    rng = np.random.default_rng(seed)
    img = np.full((200, 200), 20.0) + rng.normal(0, 2, (200, 200))
    true_corners = np.array([[50.0, 40.0], [160.0, 30.0],
                             [170.0, 150.0], [40.0, 160.0]])  # TL, TR, BR, BL
    yy, xx = np.mgrid[0:200, 0:200]
    inside = np.ones((200, 200), dtype=bool)
    for i in range(4):
        x1, y1 = true_corners[i]
        x2, y2 = true_corners[(i + 1) % 4]
        inside &= (x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1) >= 0
    img[inside] = 220.0 + rng.normal(0, 3, inside.sum())
    return np.clip(img, 0, 255), true_corners


def main():
    img, true_corners = make_photo(seed=42)
    corners = find_document_corners(img)
    warped, _ = scan_document(img, width=120, height=160)
    errors = np.linalg.norm(corners - true_corners, axis=1)
    print("True corners (TL, TR, BR, BL):")
    print(true_corners)
    print("Detected corners (TL, TR, BR, BL):")
    print(np.round(corners, 1))
    print(f"Corner error per corner (px): {np.round(errors, 2)}")
    print(f"Max corner error: {errors.max():.2f} px")
    print(f"Warped shape: {warped.shape}")
    print(f"Warped mean brightness: {warped.mean():.1f} / 255")
    print(f"Warped centre pixel: {warped[80, 60]:.1f}")
    print(f"Warped corner pixel (should be bright): {warped[0, 0]:.1f}")
    # Sanity: warping the exact true corners recovers a uniformly bright page.
    exact = warp_document(img, true_corners, 120, 160)
    print(f"Exact-corner warp mean brightness: {exact.mean():.1f} / 255")


if __name__ == "__main__":
    main()
