import numpy as np
import pytest

from clahe.clahe import apply_clahe
from clahe.demo import make_test_image


def _full_range_image(seed=0):
    """Gradient spanning the full 0-255 range plus a little noise."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:64, 0:96]
    img = (rows * 2.0 + cols * 1.5) % 256
    img = img + rng.normal(0.0, 2.0, size=img.shape)
    return np.clip(img, 0, 255)


def test_output_shape_and_dtype():
    img = np.random.default_rng(0).integers(0, 256, size=(40, 55)).astype(np.uint8)
    out = apply_clahe(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_full_range_output():
    img = _full_range_image()
    out = apply_clahe(img)
    assert out.min() <= 5
    assert out.max() >= 250


def test_local_contrast_improves_in_low_contrast_patch():
    img = make_test_image()
    out = apply_clahe(img).astype(float)
    before = img[8:24, 10:30].std()
    after = out[8:24, 10:30].std()
    assert after > before
    assert after > 2 * before


def test_clip_limit_effect():
    img = make_test_image()
    low = apply_clahe(img, clip_limit=1.0).astype(float)
    high = apply_clahe(img, clip_limit=4.0).astype(float)
    assert high.std() >= low.std()


def test_constant_image_unchanged():
    img = np.full((32, 48), 128, dtype=np.uint8)
    out = apply_clahe(img)
    np.testing.assert_array_equal(out, img)


def test_invalid_input_1d():
    with pytest.raises(ValueError, match="2D"):
        apply_clahe(np.zeros(10))


def test_invalid_input_empty():
    with pytest.raises(ValueError, match="non-empty"):
        apply_clahe(np.zeros((0, 0)))


def test_invalid_input_non_finite():
    img = np.full((16, 16), 100.0)
    img[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        apply_clahe(img)
    img[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        apply_clahe(img)


def test_tile_grid_parameter():
    img = make_test_image()
    out_2x2 = apply_clahe(img, tile_grid=(2, 2))
    out_8x8 = apply_clahe(img, tile_grid=(8, 8))
    assert out_2x2.shape == img.shape
    assert out_8x8.shape == img.shape
    assert out_2x2.dtype == np.uint8
    assert not np.array_equal(out_2x2, out_8x8)


def test_deterministic():
    img = make_test_image()
    first = apply_clahe(img)
    second = apply_clahe(img)
    np.testing.assert_array_equal(first, second)


def test_float_input_accepted():
    img = make_test_image().astype(float)
    out = apply_clahe(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
