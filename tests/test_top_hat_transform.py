"""Tests for top_hat_transform (white/black top-hat transforms)."""

import numpy as np
import pytest

from top_hat_transform.tophat import (
    black_tophat,
    closing,
    dilate,
    erode,
    opening,
    structuring_element,
    top_hat,
    white_tophat,
)


def _spot_image(height=20, width=20, bright=True):
    """Flat background with one small 3x3 bright (or dark) spot."""
    image = np.full((height, width), 100.0)
    image[8:11, 8:11] = 200.0 if bright else 10.0
    return image


def test_square_structuring_element_is_all_true():
    element = structuring_element(3, "square")
    assert element.shape == (3, 3)
    assert element.all()


def test_disk_structuring_element_shape():
    element = structuring_element(5, "disk")
    assert element.shape == (5, 5)
    assert element[2, 2]  # centre is in
    assert not element[0, 0]  # corners are out
    assert element[0, 2]  # edge midpoints are in
    assert element.sum() == 13


def test_structuring_element_rejects_even_size():
    with pytest.raises(ValueError, match="odd positive"):
        structuring_element(4, "square")


def test_structuring_element_rejects_unknown_shape():
    with pytest.raises(ValueError, match="square.*disk"):
        structuring_element(3, "triangle")


def test_erode_shrinks_bright_spot():
    image = _spot_image()
    eroded = erode(image, kernel_size=3)
    # A 3x3 bright spot eroded by 3x3 leaves a single bright pixel.
    assert eroded[9, 9] == 200.0
    assert eroded[8, 8] == 100.0
    assert eroded.max() == 200.0


def test_dilate_grows_bright_spot():
    image = _spot_image()
    dilated = dilate(image, kernel_size=3)
    assert dilated[7:12, 7:12].min() == 200.0  # spot grew by one pixel
    assert dilated[0, 0] == 100.0


def test_opening_removes_small_bright_spot():
    image = _spot_image()
    np.testing.assert_allclose(opening(image, kernel_size=5), 100.0, atol=1e-12)


def test_closing_removes_small_dark_spot():
    image = _spot_image(bright=False)
    np.testing.assert_allclose(closing(image, kernel_size=5), 100.0, atol=1e-12)


def test_opening_keeps_large_bright_region():
    image = np.full((20, 20), 100.0)
    image[5:15, 5:15] = 200.0  # 10x10 region survives a 3x3 opening
    opened = opening(image, kernel_size=3)
    assert opened[9, 9] == 200.0


def test_white_tophat_extracts_bright_spot_only():
    image = _spot_image()
    result = white_tophat(image, kernel_size=5)
    assert result[9, 9] == pytest.approx(100.0)  # spot value minus background
    assert result[0, 0] == pytest.approx(0.0)    # background cancels out
    assert result.min() >= 0.0


def test_black_tophat_extracts_dark_spot_only():
    image = _spot_image(bright=False)
    result = black_tophat(image, kernel_size=5)
    assert result[9, 9] == pytest.approx(90.0)  # background minus spot value
    assert result[0, 0] == pytest.approx(0.0)
    assert result.min() >= 0.0


def test_tophat_returns_both_transforms():
    image = _spot_image()
    white, black = top_hat(image, kernel_size=5)
    np.testing.assert_allclose(white, white_tophat(image, kernel_size=5))
    np.testing.assert_allclose(black, black_tophat(image, kernel_size=5))


def test_white_tophat_zero_on_flat_image():
    image = np.full((10, 10), 50.0)
    np.testing.assert_allclose(white_tophat(image, kernel_size=3), 0.0, atol=1e-12)
    np.testing.assert_allclose(black_tophat(image, kernel_size=3), 0.0, atol=1e-12)


def test_white_tophat_zero_on_slow_gradient():
    # A linear ramp is unchanged by opening in the interior, so the
    # top-hat is zero there; replicate padding leaves at most a 1.0
    # residue in the last column.
    image = np.tile(np.arange(20, dtype=np.float64), (20, 1))
    result = white_tophat(image, kernel_size=3)
    np.testing.assert_allclose(result[:, :-1], 0.0, atol=1e-9)
    assert result.max() <= 1.0


def test_disk_element_extracts_spot_too():
    image = _spot_image()
    result = white_tophat(image, kernel_size=5, shape="disk")
    assert result[9, 9] == pytest.approx(100.0)
    assert result[0, 0] == pytest.approx(0.0)


def test_larger_kernel_removes_larger_spot():
    image = np.full((20, 20), 100.0)
    image[7:12, 7:12] = 200.0  # 5x5 bright spot
    assert white_tophat(image, kernel_size=3)[9, 9] == pytest.approx(0.0)
    assert white_tophat(image, kernel_size=7)[9, 9] == pytest.approx(100.0)


def test_output_shape_matches_input():
    image = _spot_image(height=12, width=18)
    assert white_tophat(image).shape == image.shape
    assert black_tophat(image).shape == image.shape


def test_rejects_1d_image():
    with pytest.raises(ValueError, match="2D"):
        white_tophat(np.zeros(10))


def test_rejects_even_kernel_size():
    with pytest.raises(ValueError, match="odd positive"):
        white_tophat(np.zeros((5, 5)), kernel_size=2)


def test_rejects_unknown_shape():
    with pytest.raises(ValueError, match="square.*disk"):
        black_tophat(np.zeros((5, 5)), shape="cross")


def test_rejects_empty_image():
    with pytest.raises(ValueError, match="empty"):
        white_tophat(np.zeros((0, 0)))
