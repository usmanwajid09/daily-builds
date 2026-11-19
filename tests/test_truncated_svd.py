"""Tests for truncated SVD."""

import numpy as np
import pytest

from truncated_svd.model import TruncatedSVD


def test_reconstructs_low_rank_matrix_exactly():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(30, 2))
    v = rng.normal(size=(2, 5))
    x = u @ v
    model = TruncatedSVD(n_components=2).fit(x)
    assert model.reconstruction_error(x) < 1e-10


def test_singular_values_match_numpy_svd():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 8))
    model = TruncatedSVD(n_components=3).fit(x)
    expected = np.linalg.svd(x, compute_uv=False)[:3]
    assert np.allclose(model.singular_values_, expected, atol=1e-8)


def test_singular_values_match_numpy_svd_wide_matrix():
    # Wide matrix (more features than samples) exercises the X @ X.T path.
    rng = np.random.default_rng(2)
    x = rng.normal(size=(6, 20))
    model = TruncatedSVD(n_components=3).fit(x)
    expected = np.linalg.svd(x, compute_uv=False)[:3]
    assert np.allclose(model.singular_values_, expected, atol=1e-8)


def test_transform_matches_numpy_projection():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(25, 6))
    model = TruncatedSVD(n_components=2).fit(x)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    projected = model.transform(x)
    expected = x @ vh[:2].T
    # Signs of singular vectors are arbitrary; compare absolute values.
    assert np.allclose(np.abs(projected), np.abs(expected), atol=1e-8)


def test_explained_variance_ratio_sums_to_one_with_all_components():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 5))
    model = TruncatedSVD(n_components=5).fit(x)
    assert np.isclose(model.explained_variance_ratio_.sum(), 1.0)


def test_explained_variance_ratio_decreases():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 6))
    model = TruncatedSVD(n_components=4).fit(x)
    ratios = model.explained_variance_ratio_
    assert np.all(np.diff(ratios) <= 1e-12)


def test_fit_transform_matches_transform():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(15, 5))
    model = TruncatedSVD(n_components=2)
    projected = model.fit_transform(x)
    assert np.allclose(projected, model.transform(x))


def test_inverse_transform_roundtrip_shape():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 4))
    model = TruncatedSVD(n_components=2).fit(x)
    assert model.inverse_transform(model.transform(x)).shape == x.shape


def test_more_components_lower_reconstruction_error():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(30, 8))
    errors = [
        TruncatedSVD(n_components=k).fit(x).reconstruction_error(x)
        for k in (1, 3, 6)
    ]
    assert errors[0] > errors[1] > errors[2]


def test_transform_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        TruncatedSVD(n_components=2).transform(np.zeros((3, 4)))


def test_invalid_n_components_raises():
    with pytest.raises(ValueError, match="positive integer"):
        TruncatedSVD(n_components=0)


def test_n_components_too_large_raises():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(4, 6))
    with pytest.raises(ValueError, match="must be <="):
        TruncatedSVD(n_components=5).fit(x)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        TruncatedSVD(n_components=2).fit(np.zeros((0, 3)))


def test_1d_input_raises():
    with pytest.raises(ValueError, match="2D"):
        TruncatedSVD(n_components=2).fit(np.zeros(5))


def test_nan_input_raises():
    x = np.zeros((5, 3))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        TruncatedSVD(n_components=2).fit(x)


def test_wrong_feature_count_raises():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(10, 4))
    model = TruncatedSVD(n_components=2).fit(x)
    with pytest.raises(ValueError, match="features"):
        model.transform(rng.normal(size=(3, 5)))
