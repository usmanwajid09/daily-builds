"""Tests for face_detector_simple."""

import numpy as np
import pytest

from face_detector_simple import detect_faces, rgb_to_ycbcr, skin_mask
from face_detector_simple.demo import make_test_image


def test_demo_scene_finds_exactly_one_face():
    image, true_box = make_test_image()
    faces = detect_faces(image)
    assert len(faces) == 1
    assert faces[0] == true_box


def test_no_skin_no_faces():
    image = np.zeros((40, 40, 3))
    image[:] = (40.0, 80.0, 180.0)  # blue, no skin
    assert detect_faces(image) == []


def test_small_skin_blob_ignored():
    image = np.zeros((40, 40, 3))
    image[:] = (40.0, 80.0, 180.0)
    image[5:12, 5:12] = (210.0, 160.0, 130.0)
    assert detect_faces(image) == []


def test_rectangle_rejected_by_fill():
    image = np.zeros((60, 60, 3))
    image[:] = (40.0, 80.0, 180.0)
    image[10:40, 10:40] = (210.0, 160.0, 130.0)
    assert detect_faces(image) == []


def test_skin_mask_covers_face():
    image, true_box = make_test_image()
    mask = skin_mask(image)
    assert mask.dtype == bool
    assert mask.shape == image.shape[:2]
    top, left, bottom, right = true_box
    assert mask[30, 45]             # inside the face ellipse
    assert not mask[0, 0]           # blue background is not skin


def test_ycbcr_round_trip_shape():
    image, _ = make_test_image()
    out = rgb_to_ycbcr(image)
    assert out.shape == image.shape
    # Pure grey must have neutral chroma (Cb = Cr = 128).
    grey = np.full((4, 4, 3), 100.0)
    conv = rgb_to_ycbcr(grey)
    assert np.allclose(conv[..., 1], 128.0, atol=1e-6)
    assert np.allclose(conv[..., 2], 128.0, atol=1e-6)


def test_invalid_input_rejected():
    with pytest.raises(ValueError, match="RGB"):
        detect_faces(np.zeros((5, 5)))
    with pytest.raises(ValueError, match="RGB"):
        skin_mask(np.zeros((5, 5, 4)))


def test_two_faces_found():
    image, true_box = make_test_image()
    yy, xx = np.mgrid[0:80, 0:100]
    second = ((yy - 40) / 10.0) ** 2 + ((xx - 85) / 8.0) ** 2 <= 1.0
    image[second] = (210.0, 160.0, 130.0)
    faces = detect_faces(image)
    assert len(faces) == 2
    assert true_box in faces
