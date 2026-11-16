"""Tests for fast_corner_detector."""

import numpy as np
import pytest

from fast_corner_detector.demo import (
    SQUARE_CORNERS,
    TRIANGLE_ACUTE_CORNERS,
    make_test_image,
    make_uniform_image,
)
from fast_corner_detector.fast import (
    corner_score,
    detect_corners,
    is_corner,
)


def _near(corners, target, tol=3):
    """True if any detected corner is within ``tol`` of ``target``."""
    return any(abs(int(r) - target[0]) <= tol and abs(int(c) - target[1]) <= tol
               for r, c in corners)


def _pairwise_chebyshev(corners):
    pts = [(int(r), int(c)) for r, c in corners]
    return [max(abs(a[0] - b[0]), abs(a[1] - b[1]))
            for i, a in enumerate(pts) for b in pts[i + 1:]]


def test_square_corners_detected():
    # FAST-9 is the right setting for ordinary 90-degree corners: a perfect
    # 90-degree step corner makes at most an 11-pixel arc on the 16-circle,
    # so the stricter FAST-12 default cannot see it (see
    # test_n12_strict_on_right_angle_square).
    corners, _ = detect_corners(make_test_image(), threshold=20,
                                n_contiguous=9)
    found = sum(1 for t in SQUARE_CORNERS if _near(corners, t))
    assert found >= 3, f"only {found}/4 square corners found: {corners}"
    assert len(corners) <= 10, f"too many corners: {len(corners)}"


def test_n12_detects_acute_triangle_corners():
    # The triangle's 45-degree corners are sharp enough for FAST-12.
    corners, _ = detect_corners(make_test_image(), threshold=20)  # N=12
    found = sum(1 for t in TRIANGLE_ACUTE_CORNERS if _near(corners, t))
    assert found == 2, f"expected both acute corners, got {corners}"


def test_n12_strict_on_right_angle_square():
    # Documents FAST-12's strictness: a clean white square has no corner
    # sharp enough for a 12-pixel contiguous arc.
    img = np.zeros((80, 100), dtype=np.float64)
    img[20:60, 30:70] = 255.0
    corners, _ = detect_corners(img, threshold=20)  # N=12
    assert len(corners) == 0, f"FAST-12 should miss 90-degree corners: {corners}"


def test_uniform_image_gives_zero_corners():
    for n in (9, 12):
        corners, scores = detect_corners(make_uniform_image(), threshold=20,
                                         n_contiguous=n)
        assert len(corners) == 0
        assert scores.shape == (0,)


def test_higher_threshold_gives_fewer_or_equal_corners():
    img = make_test_image()
    counts = []
    for threshold in (5, 20, 60, 150):
        corners, _ = detect_corners(img, threshold=threshold, n_contiguous=9)
        counts.append(len(corners))
    assert counts == sorted(counts, reverse=True), (
        f"corner counts should not rise with threshold: {counts}")
    assert counts[0] > counts[-1]


def test_nms_removes_duplicate_neighbours():
    img = make_test_image()
    raw, _ = detect_corners(img, threshold=20, n_contiguous=9, nms=False)
    kept, _ = detect_corners(img, threshold=20, n_contiguous=9, nms=True,
                             nms_radius=3)
    assert len(kept) < len(raw), (
        f"NMS should collapse clusters: raw={len(raw)}, kept={len(kept)}")
    assert len(kept) >= 4
    assert all(d >= 3 for d in _pairwise_chebyshev(kept)), (
        f"two kept corners closer than the NMS radius: {kept}")


def test_invalid_input_raises_value_error():
    img = make_test_image()
    with pytest.raises(ValueError):
        detect_corners(np.zeros(10))  # 1D array
    with pytest.raises(ValueError):
        detect_corners(np.zeros((0, 0)))  # empty
    with pytest.raises(ValueError):
        detect_corners(np.zeros((0, 5)))  # empty
    with pytest.raises(ValueError):
        detect_corners(img, threshold=0)
    with pytest.raises(ValueError):
        detect_corners(img, threshold=-5)
    with pytest.raises(ValueError):
        detect_corners(img, n_contiguous=0)
    with pytest.raises(ValueError):
        detect_corners(img, n_contiguous=17)
    with pytest.raises(ValueError):
        detect_corners(img, nms_radius=0)
    with pytest.raises(ValueError):
        is_corner(img, 0, 0)  # too close to the border for the circle


def test_corners_inside_image_bounds():
    img = make_test_image()
    corners, _ = detect_corners(img, threshold=20, n_contiguous=9)
    assert len(corners) > 0
    assert corners.shape[1] == 2
    assert np.all(corners[:, 0] >= 0) and np.all(corners[:, 0] < img.shape[0])
    assert np.all(corners[:, 1] >= 0) and np.all(corners[:, 1] < img.shape[1])


def test_deterministic():
    img = make_test_image()
    c1, s1 = detect_corners(img, threshold=20, n_contiguous=9)
    c2, s2 = detect_corners(img, threshold=20, n_contiguous=9)
    assert np.array_equal(c1, c2)
    assert np.array_equal(s1, s2)


def test_border_pixels_handled_without_crash():
    # A square whose corner sits on the image edge: no crash, and nothing
    # is reported outside the image.
    img = np.zeros((30, 30), dtype=np.float64)
    img[0:15, 0:15] = 255.0
    corners, _ = detect_corners(img, threshold=20, n_contiguous=9)
    assert np.all(corners >= 0)
    # An image too small for a radius-3 circle has no interior pixels.
    tiny, _ = detect_corners(np.full((5, 5), 128.0), threshold=20)
    assert len(tiny) == 0


def test_single_pixel_helpers_agree_with_detector():
    img = make_test_image()
    assert is_corner(img, 20, 20, threshold=20, n_contiguous=9)
    assert not is_corner(img, 35, 35, threshold=20, n_contiguous=9)
    assert corner_score(img, 20, 20, threshold=20, n_contiguous=9) > 0
    assert corner_score(img, 35, 35, threshold=20, n_contiguous=9) == 0
