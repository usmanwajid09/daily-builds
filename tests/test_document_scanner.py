import numpy as np
import pytest

from document_scanner.document_scanner import (
    compute_homography,
    find_document_corners,
    order_corners,
    scan_document,
    warp_document,
)


def make_photo(seed=0, corners=None, shape=(200, 200), bg=20.0, fg=220.0):
    """Dark image with a bright convex quadrilateral; returns (img, corners)."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, bg) + rng.normal(0, 2, shape)
    if corners is None:
        corners = np.array([[50.0, 40.0], [160.0, 30.0],
                            [170.0, 150.0], [40.0, 160.0]])
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    inside = np.ones(shape, dtype=bool)
    for i in range(4):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]
        inside &= (x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1) >= 0
    img[inside] = fg + rng.normal(0, 3, inside.sum())
    return np.clip(img, 0, 255), np.asarray(corners, dtype=float)


def test_find_corners_recovers_true_corners():
    img, true = make_photo(seed=42)
    corners = find_document_corners(img)
    assert corners.shape == (4, 2)
    errors = np.linalg.norm(corners - true, axis=1)
    assert errors.max() < 3.0, f"max corner error {errors.max():.2f}px"


def test_find_corners_with_explicit_threshold():
    img, true = make_photo(seed=1)
    corners = find_document_corners(img, threshold=120.0)
    assert np.linalg.norm(corners - true, axis=1).max() < 3.0


def test_warp_output_shape_and_brightness():
    img, true = make_photo(seed=42)
    warped = warp_document(img, true, width=120, height=160)
    assert warped.shape == (160, 120)
    assert warped.mean() > 200.0  # warped page is almost all bright document


def test_scan_document_combines_detect_and_warp():
    img, true = make_photo(seed=7)
    warped, corners = scan_document(img, width=100, height=140)
    assert warped.shape == (140, 100)
    assert corners.shape == (4, 2)
    assert warped.mean() > 190.0


def test_rgb_input_works():
    img, true = make_photo(seed=3)
    rgb = np.stack([img, img, img], axis=-1)
    corners = find_document_corners(rgb)
    warped = warp_document(rgb, true, 80, 100)
    assert warped.shape == (100, 80, 3)
    assert np.linalg.norm(corners - true, axis=1).max() < 3.0


def test_homography_maps_corners():
    src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
    dst = np.array([[50.0, 40.0], [160.0, 30.0], [170.0, 150.0], [40.0, 160.0]])
    h = compute_homography(src, dst)
    mapped = (h @ np.column_stack([src, np.ones(4)]).T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    assert np.allclose(mapped, dst, atol=1e-6)


def test_order_corners_sorts_unordered_input():
    pts = np.array([[170.0, 150.0], [40.0, 160.0], [50.0, 40.0], [160.0, 30.0]])
    ordered = order_corners(pts)
    expected = np.array([[50.0, 40.0], [160.0, 30.0],
                         [170.0, 150.0], [40.0, 160.0]])
    assert np.allclose(ordered, expected)


def test_all_dark_image_raises():
    with pytest.raises(ValueError, match="no bright quadrilateral"):
        find_document_corners(np.full((100, 100), 10.0))


def test_all_bright_image_raises():
    with pytest.raises(ValueError, match="no bright quadrilateral"):
        find_document_corners(np.full((100, 100), 250.0))


def test_invalid_image_shape_raises():
    with pytest.raises(ValueError, match="must be 2D grayscale"):
        find_document_corners(np.zeros((10, 10, 10)))
    with pytest.raises(ValueError, match="must be 2D grayscale"):
        find_document_corners(np.zeros(10))


def test_invalid_corners_raise():
    img, _ = make_photo(seed=0)
    with pytest.raises(ValueError, match="shape \\(4, 2\\)"):
        warp_document(img, np.zeros((3, 2)), 100, 100)
    with pytest.raises(ValueError, match="finite"):
        warp_document(img, np.array([[0.0, np.nan]] * 4), 100, 100)
    with pytest.raises(ValueError, match="distinct"):
        warp_document(img, np.zeros((4, 2)), 100, 100)


def test_invalid_output_size_raises():
    img, true = make_photo(seed=0)
    with pytest.raises(ValueError, match="width must be positive"):
        warp_document(img, true, 0, 100)
    with pytest.raises(ValueError, match="height must be positive"):
        warp_document(img, true, 100, -5)
    with pytest.raises(ValueError, match="width must be a positive int"):
        warp_document(img, true, 100.5, 100)


def test_invalid_threshold_raises():
    img, _ = make_photo(seed=0)
    with pytest.raises(ValueError, match="threshold must be in"):
        find_document_corners(img, threshold=300.0)


def test_identity_warp_preserves_image():
    img, _ = make_photo(seed=0)
    h, w = img.shape
    corners = np.array([[0.0, 0.0], [w - 1.0, 0.0],
                        [w - 1.0, h - 1.0], [0.0, h - 1.0]])
    warped = warp_document(img, corners, w, h)
    assert np.allclose(warped, img, atol=1e-6)
