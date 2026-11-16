import numpy as np
import pytest

from texture_classifier import (
    CLASSES,
    TextureKNN,
    accuracy,
    extract_features,
    lbp_histogram,
    make_checkerboard,
    make_dots,
    make_noise,
    make_stripes,
    make_texture,
)


def _dataset(rng, n_train=20, n_test=10):
    train_X, train_y, test_X, test_y = [], [], [], []
    for name in CLASSES:
        for _ in range(n_train):
            train_X.append(make_texture(name, rng=rng))
            train_y.append(name)
        for _ in range(n_test):
            test_X.append(make_texture(name, rng=rng))
            test_y.append(name)
    return train_X, np.array(train_y), test_X, np.array(test_y)


def test_lbp_histogram_sums_to_one_and_has_256_bins():
    hist = lbp_histogram(make_checkerboard())
    assert hist.shape == (256,)
    assert hist.sum() == pytest.approx(1.0)
    assert (hist >= 0).all()


def test_constant_image_puts_all_mass_in_one_bin():
    hist = lbp_histogram(np.full((16, 16), 128.0))
    assert hist.max() == pytest.approx(1.0)
    assert np.count_nonzero(hist) == 1
    # Every neighbour equals the centre, so every bit is set: code 255.
    assert np.argmax(hist) == 255


def test_classifier_accuracy_on_synthetic_textures():
    rng = np.random.default_rng(1)
    train_X, train_y, test_X, test_y = _dataset(rng)
    clf = TextureKNN(k=3).fit(extract_features(train_X), train_y)
    assert accuracy(clf, test_X, test_y) >= 0.9


def test_different_textures_have_different_histograms():
    rng = np.random.default_rng(2)
    h_stripes = lbp_histogram(make_stripes(rng=rng))
    h_noise = lbp_histogram(make_noise(rng=rng))
    assert np.linalg.norm(h_stripes - h_noise) > 0.1


def test_make_texture_unknown_class_raises():
    with pytest.raises(ValueError):
        make_texture("plaid")


def test_invalid_orientation_raises():
    with pytest.raises(ValueError):
        make_stripes(orientation="diagonal")


def test_lbp_rejects_non_2d_image():
    with pytest.raises(ValueError):
        lbp_histogram(np.zeros((8, 8, 3)))


def test_lbp_rejects_too_small_image():
    with pytest.raises(ValueError):
        lbp_histogram(np.zeros((2, 2)))


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        TextureKNN().predict(np.zeros((1, 256)))


def test_fit_with_fewer_samples_than_k_raises():
    with pytest.raises(ValueError):
        TextureKNN(k=5).fit(np.zeros((3, 256)), ["a", "b", "c"])


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        TextureKNN(k=0)


def test_feature_dimension_mismatch_raises():
    rng = np.random.default_rng(3)
    train_X, train_y, _, _ = _dataset(rng, n_train=5, n_test=1)
    clf = TextureKNN(k=1).fit(extract_features(train_X), train_y)
    with pytest.raises(ValueError):
        clf.predict(np.zeros((1, 10)))
