import numpy as np
import pytest

from image_quality_assessor import (
    add_gaussian_noise,
    assess,
    assess_batch,
    blur_score,
    box_blur,
    brightness_score,
    contrast_score,
    estimate_noise,
    laplacian,
    laplacian_variance,
    make_checkerboard,
    make_gradient,
    make_low_contrast,
    noise_score,
    verdict_for,
)


@pytest.fixture(scope="module")
def sharp():
    return make_checkerboard(size=64, square=8)


@pytest.fixture(scope="module")
def blurred(sharp):
    return box_blur(sharp, iterations=8)


@pytest.fixture(scope="module")
def noisy(sharp):
    return add_gaussian_noise(sharp, sigma=30.0, seed=7)


@pytest.fixture(scope="module")
def low_contrast(sharp):
    return make_low_contrast(sharp)


@pytest.fixture(scope="module")
def dark(sharp):
    return sharp * 0.12


def test_assess_returns_all_keys_and_score_in_range(sharp):
    result = assess(sharp)
    for key in (
        "score", "verdict", "blur", "noise", "contrast", "brightness",
        "blur_score", "noise_score", "contrast_score", "brightness_score",
    ):
        assert key in result
    assert 0.0 <= result["score"] <= 100.0
    assert result["verdict"] in ("excellent", "good", "fair", "poor")
    for sub in ("blur_score", "noise_score", "contrast_score", "brightness_score"):
        assert 0.0 <= result[sub] <= 100.0


def test_sharp_image_scores_excellent(sharp):
    result = assess(sharp)
    assert result["score"] >= 80
    assert result["verdict"] == "excellent"


def test_verdict_boundaries():
    assert verdict_for(100.0) == "excellent"
    assert verdict_for(80.0) == "excellent"
    assert verdict_for(79.9) == "good"
    assert verdict_for(60.0) == "good"
    assert verdict_for(59.9) == "fair"
    assert verdict_for(40.0) == "fair"
    assert verdict_for(39.9) == "poor"
    assert verdict_for(0.0) == "poor"


def test_blur_ordering_sharp_beats_blurred(sharp, blurred):
    assert laplacian_variance(sharp) > laplacian_variance(blurred)
    assert blur_score(sharp) > blur_score(blurred)
    assert assess(sharp)["score"] > assess(blurred)["score"]


def test_box_blur_reduces_laplacian_variance(sharp):
    assert laplacian_variance(box_blur(sharp, iterations=1)) < laplacian_variance(sharp)
    assert laplacian_variance(box_blur(sharp, iterations=8)) < laplacian_variance(
        box_blur(sharp, iterations=1)
    )


def test_noise_ordering_noisy_beats_sharp_on_noise(sharp, noisy):
    assert estimate_noise(noisy) > estimate_noise(sharp)
    assert estimate_noise(noisy) > 10.0
    assert noise_score(noisy) < noise_score(sharp)
    assert assess(sharp)["score"] > assess(noisy)["score"]


def test_noise_estimate_recovers_sigma_for_pure_noise():
    rng = np.random.default_rng(3)
    pure = np.clip(128.0 + rng.normal(0, 25.0, (64, 64)), 0, 255)
    assert estimate_noise(pure) == pytest.approx(25.0, abs=8.0)


def test_contrast_ordering_low_contrast_scores_lower(sharp, low_contrast):
    result_sharp = assess(sharp)
    result_low = assess(low_contrast)
    assert result_low["contrast"] < result_sharp["contrast"]
    assert contrast_score(low_contrast) < contrast_score(sharp)
    assert result_low["score"] < result_sharp["score"]


def test_dark_image_scores_lower_with_low_brightness(sharp, dark):
    result_sharp = assess(sharp)
    result_dark = assess(dark)
    assert result_dark["brightness"] < 40.0
    assert brightness_score(dark) < brightness_score(sharp)
    assert result_dark["score"] < result_sharp["score"]
    assert result_dark["verdict"] in ("fair", "poor")


def test_white_image_has_low_brightness_score():
    white = np.full((64, 64), 250.0)
    assert brightness_score(white) < 30.0
    assert assess(white)["score"] < assess(make_checkerboard())["score"]


def test_overall_ordering_sharp_highest(sharp, blurred, noisy, low_contrast, dark):
    sharp_score = assess(sharp)["score"]
    for degraded in (blurred, noisy, low_contrast, dark):
        assert sharp_score > assess(degraded)["score"]


def test_rgb_matches_grayscale(sharp):
    rgb = np.stack([sharp, sharp, sharp], axis=2)
    assert assess(rgb)["score"] == pytest.approx(assess(sharp)["score"], abs=1.0)
    assert laplacian_variance(rgb) == pytest.approx(
        laplacian_variance(sharp), rel=1e-6
    )


def test_rgba_ignores_alpha(sharp):
    rgba = np.dstack([sharp, sharp, sharp, np.full((64, 64), 0.0)])
    assert assess(rgba)["score"] == pytest.approx(assess(sharp)["score"], abs=1.0)


def test_zero_to_one_range_accepted_and_matches(sharp):
    assert assess(sharp / 255.0)["score"] == pytest.approx(
        assess(sharp)["score"], abs=1.0
    )


def test_constant_image_has_zero_noise_and_contrast():
    flat = np.full((64, 64), 128.0)
    result = assess(flat)
    assert result["noise"] == pytest.approx(0.0, abs=1e-9)
    assert result["contrast"] == pytest.approx(0.0, abs=1e-9)
    assert result["blur"] == pytest.approx(0.0, abs=1e-9)
    assert result["brightness"] == pytest.approx(128.0)


def test_smooth_gradient_has_near_zero_laplacian_variance():
    grad = make_gradient()
    assert laplacian_variance(grad) < 5.0
    assert laplacian(grad).shape == grad.shape


def test_laplacian_shape_and_kernel():
    img = np.zeros((5, 5))
    img[2, 2] = 100.0
    out = laplacian(img)
    assert out.shape == (5, 5)
    assert out[2, 2] == pytest.approx(-400.0)  # -4 * centre, neighbours 0
    assert out[1, 2] == pytest.approx(100.0)  # single centre neighbour


def test_assess_batch_matches_assess(sharp, noisy):
    batch = assess_batch([sharp, noisy])
    assert len(batch) == 2
    assert batch[0]["score"] == pytest.approx(assess(sharp)["score"])
    assert batch[1]["score"] == pytest.approx(assess(noisy)["score"])


@pytest.mark.parametrize(
    "bad,msg",
    [
        (np.zeros((64,)), "1D"),
        (np.zeros((8, 8, 8, 3)), "4D"),
        (np.zeros((64, 64, 2)), "2 channels"),
        (np.zeros((0, 0)), "empty"),
        (np.zeros((2, 2)), "too small"),
        (np.full((64, 64), np.nan), "NaN"),
        (np.full((64, 64), np.inf), "inf"),
        (np.full((64, 64), -5.0), "negative"),
        (np.full((64, 64), 300.0), "above 255"),
        (np.full((64, 64), "x"), "string dtype"),
        (np.ones((64, 64), dtype=bool), "bool dtype"),
        (np.ones((64, 64), dtype=complex), "complex dtype"),
    ],
)
def test_invalid_inputs_raise_value_error(bad, msg):
    with pytest.raises(ValueError):
        assess(bad)


def test_non_array_input_raises_value_error():
    with pytest.raises(ValueError):
        assess(42)


def test_synthetic_helpers_are_deterministic_and_valid():
    first = make_checkerboard()
    second = make_checkerboard()
    assert np.array_equal(first, second)
    assert first.shape == (64, 64)
    assert first.min() >= 0.0 and first.max() <= 255.0
    noisy = add_gaussian_noise(first, sigma=10.0, seed=1)
    assert noisy.min() >= 0.0 and noisy.max() <= 255.0
