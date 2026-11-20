import numpy as np
import pytest

from road_lane_detector import (
    HORIZON_FRAC,
    detect_lanes,
    hough_lines,
    line_endpoints,
    make_road_image,
    sobel_edges,
)


def _endpoint_error(detected, truth):
    return max(abs(detected[0][0] - truth[0][0]),
               abs(detected[1][0] - truth[1][0]))


def test_detects_both_lanes_on_clean_image():
    image, truth = make_road_image()
    result = detect_lanes(image)
    assert result["left"] is not None
    assert result["right"] is not None
    assert _endpoint_error(result["left"], truth["left"]) < 10
    assert _endpoint_error(result["right"], truth["right"]) < 10


def test_detects_both_lanes_on_noisy_image():
    image, truth = make_road_image(noise=8.0, seed=2)
    result = detect_lanes(image)
    assert result["left"] is not None
    assert result["right"] is not None
    assert _endpoint_error(result["left"], truth["left"]) < 12
    assert _endpoint_error(result["right"], truth["right"]) < 12


def test_left_lane_is_left_of_right_lane():
    image, _ = make_road_image()
    result = detect_lanes(image)
    assert result["left"][0][0] < result["right"][0][0]
    assert result["left"][1][0] < result["right"][1][0]


def test_lane_endpoints_span_bottom_to_horizon():
    image, _ = make_road_image()
    result = detect_lanes(image)
    height = image.shape[0]
    for side in ("left", "right"):
        (x0, y0), (x1, y1) = result[side]
        assert y0 == height - 1
        assert y1 == pytest.approx(int(HORIZON_FRAC * height), abs=2)


def test_blank_image_detects_no_lanes():
    result = detect_lanes(np.zeros((240, 320)))
    assert result["left"] is None
    assert result["right"] is None
    assert result["votes"] == {"left": 0, "right": 0}


def test_votes_positive_when_lanes_found():
    image, _ = make_road_image()
    result = detect_lanes(image)
    assert result["votes"]["left"] >= 20
    assert result["votes"]["right"] >= 20


def test_sobel_edges_of_blank_image_are_zero():
    edges = sobel_edges(np.zeros((40, 40)))
    assert edges.shape == (40, 40)
    assert np.all(edges == 0)


def test_sobel_edges_detect_a_step():
    image = np.zeros((40, 40))
    image[:, 20:] = 255.0
    edges = sobel_edges(image)
    assert edges[:, 19].max() > 100
    assert edges[20, 5] == pytest.approx(0.0)


def test_hough_lines_finds_a_known_vertical_line():
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, 40] = True
    rho, theta, votes = hough_lines(mask, min_votes=10)
    assert abs(theta) < np.deg2rad(2)
    assert abs(rho - 40) < 2
    assert votes >= 90


def test_hough_lines_empty_mask_returns_none():
    assert hough_lines(np.zeros((50, 50), dtype=bool)) is None


def test_line_endpoints_vertical_line():
    endpoints = line_endpoints(40.0, 0.0, (100, 100))
    assert endpoints[0] == pytest.approx((40.0, 99.0))
    assert endpoints[1] == pytest.approx((40.0, 40.0))


def test_line_endpoints_horizontal_line_returns_none():
    assert line_endpoints(50.0, np.pi / 2, (100, 100)) is None


def test_detect_lanes_non_2d_raises():
    with pytest.raises(ValueError):
        detect_lanes(np.zeros((40, 40, 3)))


def test_detect_lanes_tiny_image_raises():
    with pytest.raises(ValueError):
        detect_lanes(np.zeros((20, 20)))


def test_make_road_image_shape_and_range():
    image, truth = make_road_image(width=320, height=240)
    assert image.shape == (240, 320)
    assert image.min() >= 0 and image.max() <= 255
    assert set(truth) == {"left", "right"}


def test_make_road_image_too_small_raises():
    with pytest.raises(ValueError):
        make_road_image(width=20, height=20)


def test_invalid_edge_fraction_raises():
    image, _ = make_road_image()
    with pytest.raises(ValueError):
        detect_lanes(image, edge_fraction=1.5)
