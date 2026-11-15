import numpy as np
import pytest

from hog_features.hog import HOGDescriptor


def vertical_bar(size=32):
    img = np.zeros((size, size))
    img[:, size // 2 - 2:size // 2 + 2] = 1.0
    return img


def horizontal_bar(size=32):
    img = np.zeros((size, size))
    img[size // 2 - 2:size // 2 + 2, :] = 1.0
    return img


def test_feature_length_formula():
    hog = HOGDescriptor(cell_size=8, block_size=2, n_bins=9)
    assert hog.feature_length((32, 32)) == 9 * 36  # 3x3 blocks * 4 cells * 9 bins
    assert len(hog.compute(np.zeros((32, 32)))) == 324


def test_blank_image_is_zero_vector():
    hog = HOGDescriptor()
    feats = hog.compute(np.zeros((32, 32)))
    assert np.all(feats == 0)


def test_vertical_vs_horizontal_differ():
    hog = HOGDescriptor()
    v = hog.compute(vertical_bar())
    h = hog.compute(horizontal_bar())
    assert np.linalg.norm(v - h) > 1.0


def test_identical_images_same_descriptor():
    hog = HOGDescriptor()
    a = hog.compute(vertical_bar())
    b = hog.compute(vertical_bar())
    assert np.array_equal(a, b)


def test_descriptor_values_finite_and_bounded():
    hog = HOGDescriptor()
    rng = np.random.default_rng(0)
    feats = hog.compute(rng.random((32, 32)))
    assert np.isfinite(feats).all()
    assert feats.min() >= 0
    assert feats.max() <= 1.0 + 1e-9


def test_block_normalisation():
    hog = HOGDescriptor()
    feats = hog.compute(vertical_bar()).reshape(9, 36)
    norms = np.linalg.norm(feats, axis=1)
    nonzero = norms[norms > 0]
    assert np.allclose(nonzero, 1.0, atol=1e-6)


def test_custom_params_change_length():
    hog = HOGDescriptor(cell_size=16, block_size=2, n_bins=9)
    assert len(hog.compute(np.zeros((32, 32)))) == 36
    hog2 = HOGDescriptor(cell_size=8, block_size=2, n_bins=18)
    assert len(hog2.compute(np.zeros((32, 32)))) == 648


def test_non_multiple_size_cropped():
    hog = HOGDescriptor(cell_size=8)
    feats = hog.compute(np.zeros((30, 30)))
    assert len(feats) == hog.feature_length((24, 24))


def test_invalid_cell_size_zero():
    with pytest.raises(ValueError):
        HOGDescriptor(cell_size=0)


def test_invalid_n_bins():
    with pytest.raises(ValueError):
        HOGDescriptor(n_bins=0)


def test_image_too_small_raises():
    with pytest.raises(ValueError):
        HOGDescriptor(cell_size=8).compute(np.zeros((4, 4)))


def test_colour_image_raises():
    with pytest.raises(ValueError):
        HOGDescriptor().compute(np.zeros((32, 32, 3)))


def test_nonfinite_image_raises():
    img = np.zeros((32, 32))
    img[0, 0] = np.inf
    with pytest.raises(ValueError):
        HOGDescriptor().compute(img)


def test_feature_length_too_small_raises():
    with pytest.raises(ValueError):
        HOGDescriptor(cell_size=8, block_size=2).feature_length((8, 8))
