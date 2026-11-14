import numpy as np
import pytest

from canny_edge.canny import (
    canny_edges,
    edge_pixel_count,
    gaussian_kernel_1d,
    hysteresis,
    non_maximum_suppression,
    sobel_gradients,
)
from canny_edge.demo import make_test_image


def _step_image(height=20, width=30, split=15):
    """Sharp vertical step: dark on the left, bright on the right."""
    image = np.zeros((height, width), dtype=np.float64)
    image[:, split:] = 255.0
    return image


# --- gaussian_kernel_1d -------------------------------------------------


def test_kernel_sums_to_one_and_is_symmetric():
    for sigma in (0.5, 1.0, 2.5):
        kernel = gaussian_kernel_1d(sigma)
        assert kernel.sum() == pytest.approx(1.0)
        assert np.allclose(kernel, kernel[::-1])
        assert kernel[len(kernel) // 2] == kernel.max()


def test_kernel_size_defaults_to_cover_3_sigma():
    assert len(gaussian_kernel_1d(1.0)) == 7   # 2*ceil(3)+1
    assert len(gaussian_kernel_1d(2.0)) == 13  # 2*ceil(6)+1
    assert len(gaussian_kernel_1d(1.0, kernel_size=5)) == 5


def test_invalid_sigma_raises():
    image = np.zeros((5, 5))
    for bad in (0.0, -1.0, float("nan"), float("inf"), "1.0", True):
        with pytest.raises(ValueError):
            canny_edges(image, sigma=bad)
        with pytest.raises(ValueError):
            gaussian_kernel_1d(bad)


# --- sobel_gradients ----------------------------------------------------


def test_sobel_constant_image_is_zero():
    gx, gy, magnitude = sobel_gradients(np.full((8, 8), 128.0))
    assert np.all(gx == 0.0)
    assert np.all(gy == 0.0)
    assert np.all(magnitude == 0.0)


def test_sobel_vertical_step_responds_in_gx_only():
    gx, gy, magnitude = sobel_gradients(_step_image())
    assert gx.max() > 100.0  # strong response across the step
    assert np.abs(gy).max() < 1e-9  # no horizontal edge in a vertical step
    assert np.allclose(magnitude, np.abs(gx))


# --- non_maximum_suppression --------------------------------------------


def test_nms_thins_plateau_to_one_pixel():
    # A two-row horizontal ridge with a vertical gradient: both rows are
    # equal local maxima, so exactly one row must survive.
    magnitude = np.array([[0., 0., 0., 0.],
                          [0., 3., 3., 0.],
                          [0., 3., 3., 0.],
                          [0., 0., 0., 0.]])
    gx = np.zeros_like(magnitude)
    gy = np.ones_like(magnitude)  # gradient points up/down the ridge
    thinned = non_maximum_suppression(magnitude, gx, gy)
    assert thinned.sum() == 6.0
    assert np.count_nonzero(thinned) == 2  # one pixel per column
    assert np.all(thinned[thinned > 0] == 3.0)


def test_nms_drops_non_maxima():
    # A single ridge pixel with weaker neighbours along the gradient.
    magnitude = np.array([[0., 0., 0.],
                          [0., 5., 0.],
                          [0., 2., 0.]])
    gx = np.zeros_like(magnitude)
    gy = np.ones_like(magnitude)
    thinned = non_maximum_suppression(magnitude, gx, gy)
    assert thinned[1, 1] == 5.0
    assert thinned[2, 1] == 0.0  # weaker than the pixel above it


def test_nms_rejects_mismatched_shapes():
    magnitude = np.zeros((4, 4))
    with pytest.raises(ValueError):
        non_maximum_suppression(magnitude, np.zeros((4, 3)), np.zeros((4, 4)))


# --- hysteresis ----------------------------------------------------------


def test_hysteresis_keeps_connected_weak_and_drops_isolated():
    magnitude = np.zeros((5, 7))
    magnitude[2, 1] = 10.0  # strong edge
    magnitude[2, 2] = 5.0   # weak chain leading away from it
    magnitude[2, 3] = 5.0
    magnitude[2, 4] = 5.0
    magnitude[0, 6] = 5.0   # isolated weak pixel, no strong neighbour
    edges = hysteresis(magnitude, low=4.0, high=8.0)
    assert edges.dtype == bool
    assert edges[2, 1] and edges[2, 2] and edges[2, 3] and edges[2, 4]
    assert not edges[0, 6]
    assert edges.sum() == 4


def test_hysteresis_with_no_strong_edge_keeps_nothing():
    magnitude = np.full((4, 4), 5.0)  # all weak, none strong
    assert not hysteresis(magnitude, low=4.0, high=8.0).any()


def test_hysteresis_rejects_low_above_high():
    with pytest.raises(ValueError):
        hysteresis(np.zeros((3, 3)), low=8.0, high=4.0)
    with pytest.raises(ValueError):
        hysteresis(np.zeros((3, 3)), low=-1.0, high=4.0)


# --- canny_edges ---------------------------------------------------------


def test_step_edge_is_one_pixel_wide_and_at_the_step():
    edges = canny_edges(_step_image())
    assert edges.dtype == bool
    assert edges.shape == (20, 30)
    # One column, one edge per row, right at the step.
    assert edges.sum() == 20
    assert set(np.where(edges)[1]) == {14}
    assert edge_pixel_count(edges) == 20


def test_flat_image_has_zero_edges_including_borders():
    # Replicate padding must not invent border edges; also guards the
    # floating-point residue left by Sobel on a blurred flat image.
    for value in (0.0, 0.5, 128.0, 255.0):
        edges = canny_edges(np.full((6, 9), value))
        assert not edges.any(), f"flat {value} image produced edges"
    # A single pixel and an RGB flat image have no edge either.
    assert not canny_edges(np.array([[42.0]])).any()
    assert not canny_edges(np.full((6, 9, 3), 200.0)).any()


def test_demo_image_finds_disc_and_rectangle_outlines():
    edges = canny_edges(make_test_image())
    # Disc circumference ~75 + rectangle perimeter ~72, plus a little
    # for corners and smoothing: expect a thin outline, not a blob.
    assert 100 < edges.sum() < 250
    # No edges in the far corners, away from both shapes.
    assert not edges[:5, :5].any()
    assert not edges[-5:, -5:].any()


def test_larger_sigma_suppresses_noise_more():
    rng = np.random.default_rng(0)
    noisy = rng.normal(128.0, 20.0, size=(30, 30))
    few = canny_edges(noisy, sigma=0.5).sum()
    fewer = canny_edges(noisy, sigma=2.0).sum()
    assert few > 0
    assert fewer < few


def test_rgb_matches_grayscale_for_gray_valued_image():
    gray = _step_image()
    rgb = np.stack([gray, gray, gray], axis=2)
    assert np.array_equal(canny_edges(rgb), canny_edges(gray))


def test_uint8_input_works():
    edges = canny_edges(_step_image().astype(np.uint8))
    assert edges.dtype == bool
    assert edges.sum() == 20


def test_output_shape_matches_input():
    image = np.random.default_rng(1).random((7, 11)) * 255
    edges = canny_edges(image)
    assert edges.shape == image.shape
    assert edges.dtype == bool


def test_deterministic():
    image = make_test_image()
    assert np.array_equal(canny_edges(image), canny_edges(image))


def test_low_equals_high_single_threshold_works():
    edges = canny_edges(_step_image(), low_ratio=0.3, high_ratio=0.3)
    assert edges.sum() == 20


def test_invalid_thresholds_raise():
    image = np.zeros((5, 5))
    for low, high in ((-0.1, 0.3), (0.1, 1.5), (0.4, 0.3), (0.0, 1.01)):
        with pytest.raises(ValueError):
            canny_edges(image, low_ratio=low, high_ratio=high)
    for bad in ("0.1", True, None):
        with pytest.raises(ValueError):
            canny_edges(image, low_ratio=bad)
        with pytest.raises(ValueError):
            canny_edges(image, high_ratio=bad)


def test_invalid_image_raises():
    with pytest.raises(ValueError):
        canny_edges(np.zeros((5,)))  # 1D
    with pytest.raises(ValueError):
        canny_edges(np.zeros((5, 5, 4)))  # unsupported channels
    with pytest.raises(ValueError):
        canny_edges(np.zeros((0, 5)))  # empty
    with pytest.raises(ValueError):
        canny_edges(np.full((5, 5), np.nan))  # non-finite
    with pytest.raises(ValueError):
        canny_edges(np.full((5, 5), np.inf))  # non-finite
    with pytest.raises(ValueError):
        canny_edges(np.array([["a", "b"]]))  # non-numeric
    with pytest.raises(ValueError):
        canny_edges(np.zeros((5, 5), dtype=bool))  # boolean is not numeric
