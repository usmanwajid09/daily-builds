import numpy as np
import pytest

from image_anomaly_detector import (
    PCAAnomalyDetector,
    detect_anomalies,
    extract_patches,
    train_detector,
)
from image_anomaly_detector.demo import (
    make_anomalous_image,
    make_normal_image,
)


def _normal_patches(n_images=10):
    images = [make_normal_image(seed) for seed in range(n_images)]
    return np.concatenate(
        [extract_patches(img, 8) for img in images], axis=0
    )


def _fitted_detector(n_images=10):
    return train_detector(
        [make_normal_image(seed) for seed in range(n_images)],
        patch_size=8,
        n_components=0.95,
    )


def test_extract_patches_shape_16x16_patch8_stride8():
    image = np.arange(256, dtype=np.float64).reshape(16, 16)
    patches = extract_patches(image, patch_size=8, stride=8)
    assert patches.shape == (4, 64)
    # Top-left patch is rows 0-7, cols 0-7, flattened row-major.
    assert np.array_equal(patches[0], image[0:8, 0:8].ravel())
    # Bottom-right patch is rows 8-15, cols 8-15.
    assert np.array_equal(patches[3], image[8:16, 8:16].ravel())


def test_extract_patches_stride4_count():
    image = np.zeros((16, 16))
    patches = extract_patches(image, patch_size=8, stride=4)
    # (16 - 8) // 4 + 1 = 3 positions per axis -> 9 patches.
    assert patches.shape == (9, 64)


def test_fit_reconstruct_low_rank_small_error():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 2)) @ rng.normal(size=(2, 64))
    detector = PCAAnomalyDetector(n_components=2).fit(data)
    errors = detector.reconstruction_error(data)
    assert errors.max() < 1e-10


def test_training_patch_error_below_threshold():
    detector = _fitted_detector()
    patches = _normal_patches()
    errors = detector.reconstruction_error(patches)
    assert errors.min() < detector.threshold_
    assert errors.mean() < detector.threshold_


def test_anomalous_patch_error_much_larger_than_normal():
    detector = _fitted_detector()
    normal_patch = extract_patches(make_normal_image(123), 8)[0:1]
    bright_patch = np.full((1, 64), 255.0)
    normal_error = detector.reconstruction_error(normal_patch)[0]
    anomalous_error = detector.reconstruction_error(bright_patch)[0]
    assert anomalous_error > 10 * normal_error


def test_predict_flags_anomaly_not_normal():
    detector = _fitted_detector()
    patches = _normal_patches()
    errors = detector.reconstruction_error(patches)
    # A training patch known to sit below the threshold.
    normal_patch = patches[np.argmin(errors)][None, :]
    bright_patch = np.full((1, 64), 255.0)
    flagged = detector.predict(
        np.concatenate([normal_patch, bright_patch], axis=0)
    )
    assert flagged.tolist() == [False, True]


def test_empty_patches_raise():
    with pytest.raises(ValueError):
        PCAAnomalyDetector().fit(np.empty((0, 64)))


def test_n_components_too_large_raises():
    patches = _normal_patches()
    with pytest.raises(ValueError):
        PCAAnomalyDetector(n_components=100).fit(patches)


def test_patch_size_larger_than_image_raises():
    with pytest.raises(ValueError):
        extract_patches(np.zeros((4, 4)), patch_size=8)


def test_negative_stride_raises():
    with pytest.raises(ValueError):
        extract_patches(np.zeros((16, 16)), stride=-1)


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        PCAAnomalyDetector().predict(np.zeros((1, 64)))


def test_threshold_set_after_fit():
    detector = _fitted_detector()
    assert hasattr(detector, "threshold_")
    assert detector.threshold_ > 0


def test_explained_variance_ratio_sums_sensibly():
    detector = _fitted_detector()
    total = detector.explained_variance_ratio_.sum()
    assert total >= 0.95 - 1e-9
    assert total <= 1.0 + 1e-9
    assert len(detector.explained_variance_ratio_) == detector.n_components_
