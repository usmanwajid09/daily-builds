import numpy as np
import pytest

from dog_keypoints.demo import make_test_image
from dog_keypoints.keypoints import (
    detect_keypoints,
    difference_of_gaussians,
    gaussian_kernel_1d,
    keypoint_count,
)


def test_kernel_sums_to_one_and_is_symmetric():
    kernel = gaussian_kernel_1d(1.0)
    assert kernel.sum() == pytest.approx(1.0)
    assert np.allclose(kernel, kernel[::-1])
    assert kernel[len(kernel) // 2] == kernel.max()


def test_dog_of_constant_image_is_zero():
    img = np.full((10, 10), 128.0)
    assert np.allclose(difference_of_gaussians(img), 0.0)


def test_dog_shape_and_range():
    img = make_test_image()
    dog = difference_of_gaussians(img)
    assert dog.shape == img.shape
    assert np.isfinite(dog).all()


def test_impulse_gives_keypoint_at_centre():
    img = np.zeros((21, 21))
    img[10, 10] = 255.0
    keypoints = detect_keypoints(img)
    assert len(keypoints) == 1
    row, col, response = keypoints[0]
    assert (row, col) == (10, 10)
    assert response > 5.0


def test_flat_image_has_no_keypoints():
    img = np.full((20, 20), 100.0)
    assert detect_keypoints(img) == []


def test_demo_image_has_keypoints():
    keypoints = detect_keypoints(make_test_image())
    assert len(keypoints) >= 2
    # Strongest keypoint sits on the disc or the rectangle, not in the
    # flat background.
    row, col, _ = keypoints[0]
    on_disc = (row - 30) ** 2 + (col - 25) ** 2 <= 13 ** 2
    on_rect = 10 <= row <= 24 and 50 <= col <= 70
    assert on_disc or on_rect


def test_keypoints_sorted_strongest_first():
    keypoints = detect_keypoints(make_test_image())
    responses = [item[2] for item in keypoints]
    assert responses == sorted(responses, reverse=True)


def test_higher_threshold_finds_fewer_keypoints():
    img = make_test_image()
    loose = detect_keypoints(img, contrast_threshold=1.0)
    strict = detect_keypoints(img, contrast_threshold=30.0)
    assert len(loose) >= len(strict)
    assert len(strict) >= 1


def test_max_keypoints_caps_the_list():
    img = make_test_image()
    keypoints = detect_keypoints(img, contrast_threshold=1.0, max_keypoints=3)
    assert len(keypoints) == 3


def test_keypoint_count_helper():
    assert keypoint_count(detect_keypoints(make_test_image())) >= 2
    assert keypoint_count([]) == 0


def test_no_keypoints_on_the_border():
    img = make_test_image()
    for row, col, _ in detect_keypoints(img, contrast_threshold=1.0):
        assert 0 < row < img.shape[0] - 1
        assert 0 < col < img.shape[1] - 1


def test_sigma2_must_exceed_sigma1():
    img = np.zeros((9, 9))
    with pytest.raises(ValueError):
        difference_of_gaussians(img, sigma1=1.6, sigma2=1.0)
    with pytest.raises(ValueError):
        difference_of_gaussians(img, sigma1=1.0, sigma2=1.0)


def test_invalid_sigma_raises():
    img = np.zeros((9, 9))
    with pytest.raises(ValueError):
        difference_of_gaussians(img, sigma1=0.0)
    with pytest.raises(ValueError):
        difference_of_gaussians(img, sigma1=-1.0)
    with pytest.raises(ValueError):
        difference_of_gaussians(img, sigma2=float("nan"))


def test_invalid_threshold_and_cap_raise():
    img = make_test_image()
    with pytest.raises(ValueError):
        detect_keypoints(img, contrast_threshold=-1.0)
    with pytest.raises(ValueError):
        detect_keypoints(img, max_keypoints=0)
    with pytest.raises(ValueError):
        detect_keypoints(img, max_keypoints=2.5)


def test_bad_image_raises():
    with pytest.raises(ValueError):
        detect_keypoints(np.zeros((0, 0)))
    with pytest.raises(ValueError):
        detect_keypoints(np.zeros((5, 5, 3)))
    with pytest.raises(ValueError):
        detect_keypoints(np.zeros((5, 5)) + np.nan)
