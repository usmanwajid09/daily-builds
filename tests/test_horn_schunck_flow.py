"""Tests for horn_schunck_flow (Horn-Schunck dense optical flow)."""

import numpy as np
import pytest

from horn_schunck_flow.horn_schunck_flow import (
    HornSchunckFlow,
    flow_angle,
    flow_magnitude,
    mean_flow,
    warp_image,
)


def make_blob(cy=32.0, cx=32.0, sigma=5.0, height=64, width=64):
    """Bright Gaussian blob on a dark background."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    return 255.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))


def make_pair(dx=3.0, dy=2.0):
    """Two frames: a blob translated by (dx, dy)."""
    return make_blob(), make_blob(cy=32.0 + dy, cx=32.0 + dx)


def corner_mask(shape=(64, 64), size=8):
    """Four corner blocks, far from the centred blob."""
    mask = np.zeros(shape, dtype=bool)
    mask[:size, :size] = True
    mask[:size, -size:] = True
    mask[-size:, :size] = True
    mask[-size:, -size:] = True
    return mask


# --- core behaviour -------------------------------------------------------

def test_identical_frames_give_zero_flow():
    img = make_blob()
    u, v = HornSchunckFlow().compute_flow(img, img)
    assert np.max(np.abs(u)) < 1e-10
    assert np.max(np.abs(v)) < 1e-10


def test_known_translation_recovered():
    img1, img2 = make_pair(dx=3.0, dy=2.0)
    u, v = HornSchunckFlow(alpha=1.0, n_iterations=100).compute_flow(img1, img2)
    obj = img1 > 50.0
    assert abs(float(np.mean(u[obj])) - 3.0) < 1.0
    assert abs(float(np.mean(v[obj])) - 2.0) < 1.0


def test_flow_shape_matches_input_and_is_finite():
    img1, img2 = make_pair()
    u, v = HornSchunckFlow(n_iterations=10).compute_flow(img1, img2)
    assert u.shape == img1.shape == (64, 64)
    assert v.shape == img1.shape == (64, 64)
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))


def test_background_flow_near_zero():
    img1, img2 = make_pair()
    u, v = HornSchunckFlow(alpha=1.0, n_iterations=100).compute_flow(img1, img2)
    mag = flow_magnitude(u, v)
    assert float(np.mean(mag[corner_mask()])) < 0.05


def test_determinism():
    img1, img2 = make_pair()
    flow = HornSchunckFlow(n_iterations=20)
    u1, v1 = flow.compute_flow(img1, img2)
    u2, v2 = flow.compute_flow(img1, img2)
    assert np.array_equal(u1, u2)
    assert np.array_equal(v1, v2)


def test_convergence_flow_changes_little_with_more_iterations():
    img1, img2 = make_pair()
    u100, v100 = HornSchunckFlow(n_iterations=100, tolerance=0).compute_flow(img1, img2)
    u200, v200 = HornSchunckFlow(n_iterations=200, tolerance=0).compute_flow(img1, img2)
    assert float(np.max(np.abs(u200 - u100))) < 1.0
    assert float(np.max(np.abs(v200 - v100))) < 1.0


def test_early_stop_with_tolerance():
    img1, img2 = make_pair()
    hs = HornSchunckFlow(n_iterations=100, tolerance=0.01)
    u, v = hs.compute_flow(img1, img2)
    assert 1 < hs.iterations_run_ < 100
    u_full, _ = HornSchunckFlow(n_iterations=100, tolerance=0).compute_flow(img1, img2)
    assert float(np.max(np.abs(u - u_full))) < 0.5


# --- helpers --------------------------------------------------------------

def test_flow_magnitude_known_vectors():
    u = np.array([[3.0, 0.0]])
    v = np.array([[4.0, 0.0]])
    mag = flow_magnitude(u, v)
    assert mag[0, 0] == pytest.approx(5.0)
    assert mag[0, 1] == pytest.approx(0.0)
    # Magnitude is always non-negative, even for negative components.
    rng = np.random.default_rng(0)
    assert np.all(flow_magnitude(rng.normal(size=(10, 10)), rng.normal(size=(10, 10))) >= 0)


def test_flow_angle_and_mean_flow_known_vectors():
    # 0 deg = right, 90 deg = down (image coordinates), -90 deg = up.
    assert flow_angle(np.array([[1.0]]), np.array([[0.0]]))[0, 0] == pytest.approx(0.0)
    assert flow_angle(np.array([[0.0]]), np.array([[1.0]]))[0, 0] == pytest.approx(90.0)
    assert flow_angle(np.array([[0.0]]), np.array([[-1.0]]))[0, 0] == pytest.approx(-90.0)
    assert flow_angle(np.array([[3.0]]), np.array([[4.0]]))[0, 0] == pytest.approx(
        np.degrees(np.arctan2(4, 3))
    )
    mean_u, mean_v = mean_flow(
        np.array([[1.0, 3.0], [5.0, 7.0]]), np.array([[2.0, 2.0], [2.0, 2.0]])
    )
    assert mean_u == pytest.approx(4.0)
    assert mean_v == pytest.approx(2.0)


def test_larger_alpha_gives_smoother_flow():
    img1, img2 = make_pair()

    def mean_gradient(u, v):
        gu = np.abs(np.gradient(u, axis=0)) + np.abs(np.gradient(u, axis=1))
        gv = np.abs(np.gradient(v, axis=0)) + np.abs(np.gradient(v, axis=1))
        return float(np.mean(gu + gv))

    u_small, v_small = HornSchunckFlow(alpha=0.5, n_iterations=100).compute_flow(img1, img2)
    u_large, v_large = HornSchunckFlow(alpha=10.0, n_iterations=100).compute_flow(img1, img2)
    assert mean_gradient(u_large, v_large) < mean_gradient(u_small, v_small)


def test_warp_image_identity_with_zero_flow():
    img = make_blob()
    u = np.zeros_like(img)
    v = np.zeros_like(img)
    assert np.allclose(warp_image(img, u, v), img)


def test_warp_image_recovers_first_frame():
    img1, img2 = make_pair()
    u, v = HornSchunckFlow(alpha=1.0, n_iterations=100).compute_flow(img1, img2)
    warped = warp_image(img2, u, v)
    obj = img1 > 50.0
    err_warped = float(np.mean(np.abs(warped[obj] - img1[obj])))
    err_unwarped = float(np.mean(np.abs(img2[obj] - img1[obj])))
    assert err_warped < err_unwarped


# --- input validation -----------------------------------------------------

def test_mismatched_shapes_raise():
    img1 = make_blob()
    img2 = np.zeros((32, 32))
    with pytest.raises(ValueError, match="same shape"):
        HornSchunckFlow().compute_flow(img1, img2)


def test_1d_image_raises():
    with pytest.raises(ValueError, match="2D"):
        HornSchunckFlow().compute_flow(np.zeros(10), np.zeros(10))


def test_empty_image_raises():
    with pytest.raises(ValueError, match="non-empty"):
        HornSchunckFlow().compute_flow(np.zeros((0, 0)), np.zeros((0, 0)))


def test_nan_image_raises():
    img1 = make_blob()
    img2 = make_blob()
    img2[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        HornSchunckFlow().compute_flow(img1, img2)


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        HornSchunckFlow(alpha=0)
    with pytest.raises(ValueError, match="alpha"):
        HornSchunckFlow(alpha=-1.0)


def test_invalid_n_iterations_raises():
    with pytest.raises(ValueError, match="n_iterations"):
        HornSchunckFlow(n_iterations=0)
    with pytest.raises(ValueError, match="n_iterations"):
        HornSchunckFlow(n_iterations=-5)
