import numpy as np
import pytest

from gamma_correction.correct import (
    adjust_brightness_contrast,
    contrast_stretch,
    gamma_correct,
    log_transform,
)


def test_gamma_one_returns_image_unchanged():
    img = np.array([[0, 64, 128, 192, 255]], dtype=np.float64)
    out = gamma_correct(img, 1.0)
    assert out.shape == img.shape
    assert out.dtype == np.float64
    assert np.allclose(out, img)


def test_gamma_below_one_brightens_and_above_one_darkens():
    img = np.full((10, 10), 128.0)
    assert gamma_correct(img, 0.5).mean() > img.mean()
    assert gamma_correct(img, 2.2).mean() < img.mean()
    # Exact values: 255 * (128/255) ** gamma.
    assert gamma_correct(img, 0.5).mean() == pytest.approx(
        255 * (128 / 255) ** 0.5
    )
    assert gamma_correct(img, 2.2).mean() == pytest.approx(
        255 * (128 / 255) ** 2.2
    )


def test_gamma_output_range_and_shape():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, size=(7, 11))
    out = gamma_correct(img, 2.2)
    assert out.shape == img.shape
    assert out.dtype == np.float64
    assert out.min() >= 0.0 and out.max() <= 255.0
    # Endpoints are fixed points of the power law.
    edge = np.array([[0.0, 255.0]])
    assert np.allclose(gamma_correct(edge, 0.5), edge)
    assert np.allclose(gamma_correct(edge, 2.2), edge)


def test_gamma_on_rgb_processes_all_channels():
    rgb = np.zeros((8, 8, 3))
    rgb[:, :, 0] = 64.0
    rgb[:, :, 1] = 128.0
    rgb[:, :, 2] = 192.0
    out = gamma_correct(rgb, 0.5)
    assert out.shape == rgb.shape
    assert out.dtype == np.float64
    for channel, value in enumerate((64.0, 128.0, 192.0)):
        assert out[:, :, channel].mean() == pytest.approx(
            255 * (value / 255) ** 0.5
        )
    # Darkening hits the darkest channel hardest in relative terms.
    dark = gamma_correct(rgb, 2.2)
    assert dark[:, :, 0].mean() < dark[:, :, 1].mean() < dark[:, :, 2].mean()


def test_log_transform_endpoints_and_monotonic():
    img = np.array([[0, 32, 64, 128, 200, 255]], dtype=np.float64)
    out = log_transform(img)
    assert out.shape == img.shape
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, -1] == pytest.approx(255.0)
    assert np.all(np.diff(out[0]) > 0)  # strictly increasing
    assert out.min() >= 0.0 and out.max() <= 255.0
    # Log expands the shadows: a dark pixel gains more than a bright one.
    assert out[0, 1] - img[0, 1] > out[0, -2] - img[0, -2]


def test_log_transform_general_c_endpoints():
    img = np.array([[0, 128, 255]], dtype=np.float64)
    for c in (0.5, 2.0, 10.0):
        out = log_transform(img, c=c)
        assert out[0, 0] == pytest.approx(0.0)
        assert out[0, -1] == pytest.approx(255.0)
        assert np.all(np.diff(out[0]) > 0)


def test_contrast_stretch_expands_low_contrast_image():
    rng = np.random.default_rng(1)
    img = 100.0 + rng.uniform(0, 20, size=(20, 20))  # all in [100, 120]
    out = contrast_stretch(img)
    assert out.shape == img.shape
    assert out.min() == pytest.approx(0.0, abs=1.0)
    assert out.max() == pytest.approx(255.0, abs=1.0)
    assert out.std() > img.std()


def test_contrast_stretch_flat_image_unchanged():
    img = np.full((8, 8), 77.0)
    assert np.allclose(contrast_stretch(img), img)
    rgb = np.full((8, 8, 3), 77.0)
    assert np.allclose(contrast_stretch(rgb), rgb)


def test_brightness_plus_30_raises_mean_by_30():
    img = np.full((10, 10), 120.0)
    out = adjust_brightness_contrast(img, brightness=30.0)
    assert out.mean() == pytest.approx(150.0)
    # Negative brightness lowers the mean by the same amount.
    assert adjust_brightness_contrast(img, brightness=-30.0).mean() == pytest.approx(90.0)


def test_brightness_contrast_identity():
    img = np.random.default_rng(2).uniform(0, 255, size=(9, 9))
    out = adjust_brightness_contrast(img, brightness=0.0, contrast=1.0)
    assert np.allclose(out, img)


def test_contrast_above_one_spreads_values():
    img = np.array([[100.0, 127.5, 155.0]])
    out = adjust_brightness_contrast(img, contrast=2.0)
    assert out[0, 0] < img[0, 0]
    assert out[0, 1] == pytest.approx(127.5)  # midpoint is a fixed point
    assert out[0, 2] > img[0, 2]
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_rejects_bad_gamma():
    img = np.zeros((5, 5))
    for bad in (0.0, -1.0, float("nan"), float("inf"), "2.2", True):
        with pytest.raises(ValueError):
            gamma_correct(img, bad)


def test_rejects_bad_log_c_and_negative_contrast():
    img = np.zeros((5, 5))
    for bad in (0.0, -2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            log_transform(img, c=bad)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            adjust_brightness_contrast(img, contrast=bad)
    with pytest.raises(ValueError):
        adjust_brightness_contrast(img, brightness=float("nan"))


def test_rejects_bad_image_input():
    with pytest.raises(ValueError):
        gamma_correct(np.zeros((5,)), gamma=1.0)  # 1D
    with pytest.raises(ValueError):
        gamma_correct(np.zeros((0, 5)), gamma=1.0)  # empty
    with pytest.raises(ValueError):
        gamma_correct(np.full((5, 5), np.nan), gamma=1.0)  # non-finite
    with pytest.raises(ValueError):
        gamma_correct(np.array([["a", "b"]]), gamma=1.0)  # non-numeric
    with pytest.raises(ValueError):
        gamma_correct(np.zeros((5, 5, 4)), gamma=1.0)  # unsupported channels
    with pytest.raises(ValueError):
        log_transform(np.zeros((5,)))  # 1D rejected by the others too
    with pytest.raises(ValueError):
        contrast_stretch(np.zeros((0, 0)))


def test_rejects_bad_percentile_range():
    img = np.zeros((5, 5))
    for low, high in ((50, 50), (98, 2), (-1, 98), (2, 101), (0, 0)):
        with pytest.raises(ValueError):
            contrast_stretch(img, low=low, high=high)
    # Boundary values are fine.
    assert contrast_stretch(img, low=0, high=100).shape == img.shape


def test_float_01_input_is_scaled_to_255():
    img = np.full((4, 4), 0.5)  # 0.5 in [0, 1] == 127.5 in [0, 255]
    out = gamma_correct(img, 1.0)
    assert out.mean() == pytest.approx(127.5)
    assert np.allclose(log_transform(np.array([[0.0, 1.0]])), [[0.0, 255.0]])
