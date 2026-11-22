"""Tests for the simplified SIFT feature detector/descriptor."""

import numpy as np
import pytest

from sift_features.sift_features import (
    SIFTFeatures,
    gaussian_kernel_1d,
    match_descriptors,
)
from sift_features.demo import make_test_image


def _blank(height=64, width=64):
    return np.zeros((height, width), dtype=np.float64)


def _gaussian_blob(height=64, width=64, centre=(32, 32), sigma=5.0):
    rows, cols = np.mgrid[0:height, 0:width]
    r2 = (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2
    return np.exp(-r2 / (2 * sigma ** 2))


def _shifted(image, dy=5, dx=7):
    return np.roll(np.roll(image, dy, axis=0), dx, axis=1)


# --- detection ---------------------------------------------------------------

def test_blank_image_gives_zero_keypoints():
    sift = SIFTFeatures()
    assert sift.detect(_blank()) == []


def test_bright_blob_gives_at_least_one_keypoint():
    sift = SIFTFeatures()
    keypoints = sift.detect(_gaussian_blob())
    assert len(keypoints) >= 1


def test_keypoint_coordinates_within_image_bounds():
    image = make_test_image()
    sift = SIFTFeatures()
    keypoints = sift.detect(image)
    assert keypoints
    for kp in keypoints:
        assert 0 <= kp["y"] < image.shape[0]
        assert 0 <= kp["x"] < image.shape[1]
        assert kp["scale"] > 0
        assert -np.pi <= kp["orientation"] <= np.pi


# --- descriptors ---------------------------------------------------------------

def test_descriptor_shape_is_n_by_128():
    image = make_test_image()
    sift = SIFTFeatures()
    keypoints, descriptors = sift.detect_and_describe(image)
    assert descriptors.shape == (len(keypoints), 128)


def test_descriptors_have_unit_norm():
    _, descriptors = SIFTFeatures().detect_and_describe(make_test_image())
    assert descriptors.shape[0] > 0
    norms = np.linalg.norm(descriptors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_descriptor_values_finite_and_bounded():
    _, descriptors = SIFTFeatures().detect_and_describe(make_test_image())
    assert np.all(np.isfinite(descriptors))
    assert descriptors.min() >= 0.0
    assert descriptors.max() <= 1.0


def test_identical_images_give_identical_descriptors():
    image = make_test_image()
    sift = SIFTFeatures()
    _, desc1 = sift.detect_and_describe(image)
    _, desc2 = sift.detect_and_describe(image)
    np.testing.assert_allclose(desc1, desc2, atol=1e-12)


def test_determinism_same_image_twice():
    image = make_test_image()
    sift = SIFTFeatures()
    kps1, desc1 = sift.detect_and_describe(image)
    kps2, desc2 = sift.detect_and_describe(image)
    assert [(kp["y"], kp["x"]) for kp in kps1] == \
        [(kp["y"], kp["x"]) for kp in kps2]
    np.testing.assert_allclose(desc1, desc2, atol=1e-12)


# --- matching ------------------------------------------------------------------

def test_shifted_image_matches():
    image = make_test_image()
    sift = SIFTFeatures()
    kps1, desc1 = sift.detect_and_describe(image)
    kps2, desc2 = sift.detect_and_describe(_shifted(image))
    matches = match_descriptors(desc1, desc2, ratio=0.75)
    assert len(matches) >= 1


def test_shifted_matches_are_close_in_position():
    image = make_test_image()
    sift = SIFTFeatures()
    kps1, desc1 = sift.detect_and_describe(image)
    kps2, desc2 = sift.detect_and_describe(_shifted(image, dy=5, dx=7))
    matches = match_descriptors(desc1, desc2, ratio=0.75)
    assert matches
    for i, j, _ in matches:
        assert abs((kps2[j]["y"] - kps1[i]["y"]) - 5) <= 2.0
        assert abs((kps2[j]["x"] - kps1[i]["x"]) - 7) <= 2.0


def test_ratio_test_rejects_ambiguous_matches():
    rng = np.random.default_rng(0)
    _, descriptors = SIFTFeatures().detect_and_describe(make_test_image())
    random_desc = rng.random((20, 128))
    random_desc /= np.linalg.norm(random_desc, axis=1, keepdims=True)
    matches = match_descriptors(descriptors, random_desc, ratio=0.75)
    assert len(matches) == 0


# --- validation ------------------------------------------------------------------

@pytest.mark.parametrize("bad_image", [
    np.zeros(64),                       # 1-D
    np.zeros((0, 0)),                   # empty
    np.zeros((16, 16)),                 # too small
    np.full((64, 64), np.nan),          # NaN
])
def test_invalid_images_raise_value_error(bad_image):
    with pytest.raises(ValueError):
        SIFTFeatures().detect(bad_image)


def test_bad_ratio_raises_value_error():
    desc = np.eye(128)[:2]
    with pytest.raises(ValueError):
        match_descriptors(desc, desc, ratio=0.0)
    with pytest.raises(ValueError):
        match_descriptors(desc, desc, ratio=1.5)


def test_bad_n_octaves_raises_value_error():
    with pytest.raises(ValueError):
        SIFTFeatures(n_octaves=0)
    with pytest.raises(ValueError):
        SIFTFeatures(n_octaves=-1)


# --- helpers ---------------------------------------------------------------------

def test_gaussian_kernel_is_normalised_and_symmetric():
    kernel = gaussian_kernel_1d(1.6)
    assert kernel.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(kernel, kernel[::-1])
