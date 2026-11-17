"""Tests for optical_flow."""

import numpy as np
import pytest

from optical_flow import dense_flow, lucas_kanade, spatial_gradients
from optical_flow.demo import make_frames


def test_zero_motion_gives_zero_flow():
    prev, _ = make_frames()
    flow = lucas_kanade(prev, prev, [(40, 30), (20, 20)], window_size=15)
    assert flow.shape == (2, 2)
    assert np.allclose(flow, 0.0, atol=1e-9)


def test_recovers_known_translation():
    prev, curr = make_frames(dx=3, dy=2)
    flow = lucas_kanade(prev, curr, [(40, 30)], window_size=15)
    assert abs(flow[0, 0] - 3.0) < 0.75
    assert abs(flow[0, 1] - 2.0) < 0.75


def test_recovers_translation_at_multiple_points():
    prev, curr = make_frames(dx=3, dy=2)
    flow = lucas_kanade(prev, curr, [(35, 25), (45, 35), (40, 30)], window_size=15)
    assert flow.shape == (3, 2)
    assert np.allclose(flow, [3.0, 2.0], atol=0.75)


def test_flat_region_gives_zero_flow():
    flat = np.full((40, 50), 100.0)
    moved = np.full((40, 50), 100.0)
    flow = lucas_kanade(flat, moved, [(25, 20)], window_size=15)
    assert np.allclose(flow, 0.0)


def test_aperture_problem_gives_zero_flow():
    # A straight vertical edge moving right: flow along the edge is
    # unobservable, so the gradient matrix is singular -> (0, 0).
    prev = np.zeros((40, 50))
    prev[:, 25:] = 100.0
    curr = np.zeros((40, 50))
    curr[:, 27:] = 100.0
    flow = lucas_kanade(prev, curr, [(25, 20)], window_size=15)
    assert np.allclose(flow, 0.0)


def test_shape_mismatch_raises():
    prev, curr = make_frames()
    with pytest.raises(ValueError):
        lucas_kanade(prev, curr[:30], [(40, 30)])


def test_input_validation():
    prev, curr = make_frames()
    with pytest.raises(ValueError):
        lucas_kanade(np.zeros(10), curr, [(40, 30)])
    with pytest.raises(ValueError):
        lucas_kanade(prev, np.zeros((5, 5, 3)), [(40, 30)])
    with pytest.raises(ValueError):
        lucas_kanade(prev, curr, [(40, 30)], window_size=10)
    with pytest.raises(ValueError):
        lucas_kanade(prev, curr, [(40, 30)], window_size=2)
    with pytest.raises(ValueError):
        lucas_kanade(prev, curr, [(40, 30)], min_eigenvalue=-1)
    with pytest.raises(ValueError):
        lucas_kanade(prev, curr, [])
    with pytest.raises(ValueError):
        lucas_kanade(prev, curr, [(100, 30)])
    with pytest.raises(ValueError):
        lucas_kanade(prev, curr, [(40, 100)])
    with pytest.raises(ValueError):
        lucas_kanade(prev, curr, [(40, 30, 0)])


def test_spatial_gradients():
    # A ramp increasing along x has Ix = 1, Iy = 0 everywhere inside.
    image = np.tile(np.arange(10, dtype=float), (6, 1))
    ix, iy = spatial_gradients(image)
    assert ix.shape == image.shape
    assert iy.shape == image.shape
    assert np.allclose(ix[:, 1:-1], 1.0)
    assert np.allclose(iy, 0.0)
    with pytest.raises(ValueError):
        spatial_gradients(np.zeros((4, 4, 3)))


def test_dense_flow_shape_and_values():
    prev, curr = make_frames(dx=3, dy=2)
    u, v, xs, ys = dense_flow(prev, curr, window_size=15, step=10)
    assert u.shape == (len(ys), len(xs))
    assert v.shape == u.shape
    # The grid point nearest the blob centre should see ~(3, 2).
    cy = int(np.argmin(np.abs(ys - 30)))
    cx = int(np.argmin(np.abs(xs - 40)))
    assert abs(u[cy, cx] - 3.0) < 0.75
    assert abs(v[cy, cx] - 2.0) < 0.75
    # Far from the blob the image is flat, so flow is zero there.
    assert u[0, 0] == 0.0 and v[0, 0] == 0.0


def test_dense_flow_validation():
    prev, curr = make_frames()
    with pytest.raises(ValueError):
        dense_flow(prev, curr, window_size=15, step=0)
    with pytest.raises(ValueError):
        dense_flow(np.zeros((5, 5)), np.zeros((5, 5)), window_size=15)
