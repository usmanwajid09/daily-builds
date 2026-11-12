import numpy as np
import pytest

from edge_detector.demo import make_test_image
from edge_detector.edges import (
    detect_edges,
    edge_pixel_count,
    sobel_gradients,
    to_grayscale,
)


def _vertical_step():
    """Left half 0, right half 255: one vertical edge at column 5."""
    img = np.zeros((10, 10))
    img[:, 5:] = 255.0
    return img


def _horizontal_step():
    img = np.zeros((10, 10))
    img[5:, :] = 255.0
    return img


def test_constant_image_has_no_edges():
    assert not detect_edges(np.full((8, 8), 128.0)).any()
    assert not detect_edges(np.zeros((8, 8))).any()


def test_vertical_step_detected_at_boundary():
    edges = detect_edges(_vertical_step(), threshold=0.1)
    # The edge band sits on the step; columns far from it stay clear.
    assert edges[:, 4:6].any()
    assert not edges[:, 0].any()
    assert not edges[:, 9].any()


def test_horizontal_step_detected_at_boundary():
    edges = detect_edges(_horizontal_step(), threshold=0.1)
    assert edges[4:6, :].any()
    assert not edges[0, :].any()
    assert not edges[9, :].any()


def test_output_shape_matches_input():
    img = np.random.default_rng(0).random((7, 11)) * 255
    edges = detect_edges(img)
    assert edges.shape == img.shape
    assert edges.dtype == bool


def test_threshold_0_keeps_all_nonzero_gradients():
    _, _, mag = sobel_gradients(_vertical_step())
    edges = detect_edges(_vertical_step(), threshold=0.0, nms=False)
    assert edges.sum() == int((mag > 0).sum())


def test_threshold_1_keeps_only_strongest():
    edges = detect_edges(_vertical_step(), threshold=1.0)
    assert edges.any()
    # Only the peak-magnitude pixels survive at threshold 1.
    _, _, mag = sobel_gradients(_vertical_step())
    assert edges.sum() <= int((mag == mag.max()).sum())


def test_nms_thins_edges():
    raw = detect_edges(_vertical_step(), threshold=0.1, nms=False)
    thin = detect_edges(_vertical_step(), threshold=0.1, nms=True)
    assert thin.sum() <= raw.sum()
    assert thin.any()


def test_rgb_input_converted_to_grayscale():
    rgb = np.zeros((10, 10, 3))
    rgb[:, 5:, :] = 255.0
    edges = detect_edges(rgb, threshold=0.1)
    assert edges.shape == (10, 10)
    assert edges[:, 4:6].any()


def test_grayscale_conversion_values():
    gray = to_grayscale(np.array([[[255, 255, 255], [0, 0, 0]]]))
    assert gray.shape == (1, 2)
    assert gray[0, 0] == pytest.approx(255.0)
    assert gray[0, 1] == pytest.approx(0.0)


def test_demo_image_has_sane_edge_count():
    img = make_test_image()
    edges = detect_edges(img, threshold=0.15)
    count = edge_pixel_count(edges)
    # Disc circumference (~2*pi*12 = 75) + rectangle perimeter (~84),
    # thinned to one pixel wide: expect roughly 100-250 edge pixels.
    assert 80 <= count <= 300


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        detect_edges(np.zeros((5,)))  # 1D
    with pytest.raises(ValueError):
        detect_edges(np.zeros((0, 5)))  # empty
    with pytest.raises(ValueError):
        detect_edges(np.full((5, 5), np.nan))  # non-finite
    with pytest.raises(ValueError):
        detect_edges(np.array([["a", "b"]]))  # non-numeric
    with pytest.raises(ValueError):
        detect_edges(np.zeros((5, 5, 4)))  # unsupported channels
    with pytest.raises(ValueError):
        detect_edges(np.zeros((5, 5)), threshold=1.5)
    with pytest.raises(ValueError):
        detect_edges(np.zeros((5, 5)), threshold=-0.1)


def test_edge_pixel_count_requires_bool_map():
    with pytest.raises(ValueError):
        edge_pixel_count(np.zeros((3, 3)))
    assert edge_pixel_count(np.array([[True, False], [False, True]])) == 2
