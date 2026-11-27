"""Tests for hdr_merge."""
import numpy as np
import pytest

from hdr_merge import hat_weight, merge_hdr, psnr, reinhard_tone_map
from hdr_merge.demo import build_scene, expose


def make_stack(times=(0.25, 1.0, 4.0, 16.0)):
    scene = build_scene()
    exposures = [expose(scene, t) for t in times]
    return scene, exposures, list(times)


def test_merge_recovers_scene_log_correlation():
    scene, exposures, times = make_stack()
    hdr = merge_hdr(exposures, times)
    corr = np.corrcoef(np.log(hdr.ravel()), np.log(scene.ravel()))[0, 1]
    assert corr > 0.99


def test_merge_output_shape_dtype():
    _, exposures, times = make_stack()
    hdr = merge_hdr(exposures, times)
    assert hdr.shape == exposures[0].shape
    assert hdr.dtype == np.float64
    assert np.all(np.isfinite(hdr))
    assert (hdr > 0).all()


def test_merge_dynamic_range_exceeds_single_exposure():
    _, exposures, times = make_stack()
    hdr = merge_hdr(exposures, times)
    # The lamp (bright) and the dark background are both recovered.
    assert hdr[36, 48].mean() > 10 * hdr[0, 0].mean()
    assert hdr[36, 48].mean() > 1.0


def test_hat_weight_endpoints_and_peak():
    assert hat_weight(0.0) == 0.0
    assert hat_weight(255.0) == 0.0
    assert hat_weight(127.5) == 127.5
    assert hat_weight(64.0) == 64.0
    assert hat_weight(200.0) == 55.0


def test_hat_weight_array():
    weights = hat_weight(np.array([0, 64, 128, 255], dtype=np.float64))
    assert np.allclose(weights, [0, 64, 127, 0])


def test_grayscale_exposures_accepted():
    gray = np.full((16, 16), 100, dtype=np.uint8)
    exposures = [np.clip(gray.astype(float) * t, 0, 255).astype(np.uint8)
                 for t in (0.5, 2.0)]
    hdr = merge_hdr(exposures, [0.5, 2.0])
    assert hdr.shape == (16, 16, 3)


def test_tone_map_output_uint8_range():
    _, exposures, times = make_stack()
    hdr = merge_hdr(exposures, times)
    out = reinhard_tone_map(hdr)
    assert out.shape == hdr.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_tone_map_white_point_maps_near_white():
    radiance = np.full((8, 8, 3), 4.0)
    out = reinhard_tone_map(radiance, white_point=4.0)
    assert out.max() >= 250


def test_tone_map_dark_stays_dark():
    radiance = np.full((8, 8, 3), 0.01)
    out = reinhard_tone_map(radiance, white_point=4.0)
    assert out.max() < 30


def test_tone_map_gamma_brightens():
    _, exposures, times = make_stack()
    hdr = merge_hdr(exposures, times)
    plain = reinhard_tone_map(hdr, gamma=1.0)
    bright = reinhard_tone_map(hdr, gamma=2.2)
    assert bright.mean() > plain.mean()


def test_tone_map_grayscale_input():
    radiance = np.full((8, 8), 1.0)
    out = reinhard_tone_map(radiance)
    assert out.shape == (8, 8)
    assert out.dtype == np.uint8


def test_psnr_identical_is_inf():
    img = np.zeros((4, 4, 3))
    assert psnr(img, img) == float("inf")


def test_merge_needs_two_exposures():
    img = np.full((8, 8, 3), 100, dtype=np.uint8)
    with pytest.raises(ValueError):
        merge_hdr([img], [1.0])


def test_merge_mismatched_times_raises():
    img = np.full((8, 8, 3), 100, dtype=np.uint8)
    with pytest.raises(ValueError):
        merge_hdr([img, img], [1.0])


def test_merge_nonpositive_time_raises():
    img = np.full((8, 8, 3), 100, dtype=np.uint8)
    with pytest.raises(ValueError):
        merge_hdr([img, img], [1.0, 0.0])
    with pytest.raises(ValueError):
        merge_hdr([img, img], [1.0, -2.0])


def test_merge_shape_mismatch_raises():
    a = np.full((8, 8, 3), 100, dtype=np.uint8)
    b = np.full((8, 9, 3), 100, dtype=np.uint8)
    with pytest.raises(ValueError):
        merge_hdr([a, b], [1.0, 2.0])


def test_tone_map_invalid_args_raise():
    with pytest.raises(ValueError):
        reinhard_tone_map(np.full((4, 4), 1.0), white_point=0)
    with pytest.raises(ValueError):
        reinhard_tone_map(np.full((4, 4), 1.0), gamma=0)
    with pytest.raises(ValueError):
        reinhard_tone_map(np.full((4, 4), -1.0))
    with pytest.raises(ValueError):
        reinhard_tone_map(np.array([], dtype=np.float64))


def test_merge_deterministic():
    _, exposures, times = make_stack()
    assert np.array_equal(merge_hdr(exposures, times),
                          merge_hdr(exposures, times))
