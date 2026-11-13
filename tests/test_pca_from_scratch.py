import numpy as np
import pytest

from pca_from_scratch.model import PCA


def _correlated():
    rng = np.random.default_rng(0)
    n = 100
    factor = rng.normal(size=n)
    x = np.column_stack([
        3.0 * factor + rng.normal(scale=0.2, size=n),
        3.0 * factor + rng.normal(scale=0.2, size=n),
        rng.normal(scale=1.0, size=n),
    ])
    return x


def test_first_component_captures_correlated_direction():
    x = _correlated()
    model = PCA(n_components=1).fit(x)
    # First component should point mostly along features 0 and 1.
    comp = model.components_[0]
    assert abs(comp[0]) > 0.6 and abs(comp[1]) > 0.6
    assert abs(comp[2]) < 0.5


def test_explained_variance_ratio_sums_to_one_for_all_components():
    x = _correlated()
    model = PCA().fit(x)
    assert np.sum(model.explained_variance_ratio_) == pytest.approx(1.0)


def test_ratios_are_descending():
    x = _correlated()
    model = PCA().fit(x)
    ratios = model.explained_variance_ratio_
    assert np.all(np.diff(ratios) <= 0)


def test_full_reconstruction_is_exact():
    x = _correlated()
    model = PCA().fit(x)
    reconstructed = model.inverse_transform(model.transform(x))
    assert np.allclose(reconstructed, x)


def test_transform_shape():
    x = _correlated()
    model = PCA(n_components=2).fit(x)
    projected = model.transform(x)
    assert projected.shape == (100, 2)


def test_fit_transform_matches_fit_then_transform():
    x = _correlated()
    a = PCA(n_components=2).fit_transform(x)
    b = PCA(n_components=2).fit(x).transform(x)
    assert np.allclose(np.abs(a), np.abs(b))  # sign of a component may flip


def test_low_rank_data_keeps_most_variance_in_one_component():
    x = _correlated()
    model = PCA(n_components=1).fit(x)
    assert model.explained_variance_ratio_[0] > 0.8


def test_rejects_bad_n_components():
    with pytest.raises(ValueError):
        PCA(n_components=0)
    with pytest.raises(ValueError):
        PCA(n_components=-1)


def test_rejects_more_components_than_features():
    with pytest.raises(ValueError):
        PCA(n_components=4).fit(np.zeros((10, 3)))


def test_rejects_single_sample():
    with pytest.raises(ValueError):
        PCA().fit(np.array([[1.0, 2.0, 3.0]]))


def test_rejects_constant_data():
    with pytest.raises(ValueError):
        PCA().fit(np.ones((10, 3)))


def test_rejects_nan_values():
    with pytest.raises(ValueError):
        PCA().fit(np.array([[0.0, np.nan], [1.0, 1.0]]))


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        PCA().transform(np.array([[0.0, 0.0, 0.0]]))


def test_rejects_wrong_feature_count():
    x = _correlated()
    model = PCA(n_components=2).fit(x)
    with pytest.raises(ValueError):
        model.transform(np.array([[0.0, 0.0]]))
