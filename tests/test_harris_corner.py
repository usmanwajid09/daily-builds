import numpy as np
import pytest

from harris_corner.corners import detect_corners, harris_response
from harris_corner.demo import make_test_image

_RECT_CORNERS = [(15, 20), (15, 59), (44, 20), (44, 59)]


def _near(corners, target, tol=2):
    """True if any detected corner is within ``tol`` of ``target``."""
    return any(abs(r - target[0]) <= tol and abs(c - target[1]) <= tol
               for r, c in corners)


def test_rectangle_has_four_corners():
    corners = detect_corners(make_test_image(), max_corners=10)
    assert len(corners) == 4
    for target in _RECT_CORNERS:
        assert _near(corners, target), f"no corner near {target}: {corners}"


def test_flat_image_has_no_corners():
    assert detect_corners(np.full((20, 20), 128.0)) == []
    assert detect_corners(np.zeros((20, 20))) == []


def test_response_shape_matches_input():
    img = np.random.default_rng(0).random((7, 11)) * 255
    response = harris_response(img)
    assert response.shape == img.shape
    assert response.dtype == np.float64


def test_corners_are_strongest_first():
    img = make_test_image()
    corners = detect_corners(img, max_corners=10)
    response = harris_response(img)
    scores = [response[r, c] for r, c in corners]
    assert scores == sorted(scores, reverse=True)


def test_max_corners_is_respected():
    corners = detect_corners(make_test_image(), max_corners=2)
    assert len(corners) == 2


def test_edge_only_image_has_no_corners():
    # A single straight step edge: gradients in one direction only,
    # so the Harris response should stay <= 0 everywhere.
    img = np.zeros((20, 20))
    img[:, 10:] = 255.0
    assert detect_corners(img) == []
    assert harris_response(img).max() <= 0


def test_single_corner_found():
    # An L-shape: one bright square in the top-left of a black image.
    img = np.zeros((30, 30))
    img[:15, :15] = 255.0
    corners = detect_corners(img, max_corners=5)
    assert len(corners) == 1
    assert _near(corners, (14, 14), tol=2)


def test_rgb_input_works():
    rgb = np.zeros((60, 80, 3))
    rgb[15:45, 20:60, :] = 255.0
    corners = detect_corners(rgb, max_corners=10)
    assert len(corners) == 4
    for target in _RECT_CORNERS:
        assert _near(corners, target)


def test_higher_threshold_keeps_fewer_corners():
    img = make_test_image()
    many = detect_corners(img, threshold_rel=0.01)
    few = detect_corners(img, threshold_rel=0.99)
    assert len(few) <= len(many)
    assert len(few) >= 1


def test_invalid_inputs_raise():
    img = np.zeros((10, 10))
    with pytest.raises(ValueError):
        detect_corners(np.zeros((5,)))  # 1D
    with pytest.raises(ValueError):
        detect_corners(np.zeros((0, 5)))  # empty
    with pytest.raises(ValueError):
        detect_corners(np.full((5, 5), np.nan))  # non-finite
    with pytest.raises(ValueError):
        detect_corners(img, max_corners=0)
    with pytest.raises(ValueError):
        detect_corners(img, k=0.0)
    with pytest.raises(ValueError):
        detect_corners(img, k=1.0)
    with pytest.raises(ValueError):
        detect_corners(img, threshold_rel=1.5)
    with pytest.raises(ValueError):
        detect_corners(img, threshold_rel=-0.1)
    with pytest.raises(ValueError):
        detect_corners(img, window_radius=-1)
    with pytest.raises(ValueError):
        detect_corners(img, nms_radius=-1)
