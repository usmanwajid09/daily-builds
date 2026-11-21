"""Tests for mlp_autoencoder."""

import numpy as np
import pytest

from mlp_autoencoder import MLPAutoencoder
from mlp_autoencoder.demo import make_data


def test_rejects_bad_dimensions():
    with pytest.raises(ValueError, match="positive integer"):
        MLPAutoencoder(input_dim=0)
    with pytest.raises(ValueError, match="positive integer"):
        MLPAutoencoder(input_dim=8, hidden_dim=0)
    with pytest.raises(ValueError, match="bottleneck"):
        MLPAutoencoder(input_dim=4, latent_dim=4)
    with pytest.raises(ValueError, match="bottleneck"):
        MLPAutoencoder(input_dim=4, latent_dim=8)


def test_rejects_bad_training_data():
    model = MLPAutoencoder(input_dim=4, hidden_dim=8, latent_dim=2)
    with pytest.raises(ValueError, match="2D"):
        model.fit(np.zeros((3, 4, 1)))
    with pytest.raises(ValueError, match="empty"):
        model.fit(np.zeros((0, 4)))
    with pytest.raises(ValueError, match="finite"):
        model.fit(np.array([[1.0, np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="features"):
        model.fit(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="positive"):
        model.fit(np.zeros((20, 4)), epochs=0)
    with pytest.raises(ValueError, match="positive"):
        model.fit(np.zeros((20, 4)), lr=0.0)


def test_not_fitted_raises():
    model = MLPAutoencoder(input_dim=4, hidden_dim=8, latent_dim=2)
    X = np.zeros((3, 4))
    for call in (model.encode, model.decode, model.reconstruct,
                 model.reconstruction_error, model.anomaly_score):
        with pytest.raises(RuntimeError, match="not fitted"):
            call(X)


def test_fit_reduces_loss():
    X = make_data(200)
    model = MLPAutoencoder(input_dim=8, hidden_dim=16, latent_dim=2, seed=0)
    out0, _ = model._forward(X)
    before = float(np.mean((out0 - X) ** 2))
    model.fit(X, epochs=200, lr=0.01)
    assert len(model.history_) == 200
    assert model.history_[-1] < before / 10
    assert model.history_[-1] < model.history_[0]


def test_encode_decode_shapes():
    X = make_data(50)
    model = MLPAutoencoder(input_dim=8, hidden_dim=16, latent_dim=2, seed=1)
    model.fit(X, epochs=50, lr=0.01)
    codes = model.encode(X)
    assert codes.shape == (50, 2)
    recon = model.decode(codes)
    assert recon.shape == X.shape
    assert np.allclose(recon, model.reconstruct(X), atol=1e-9)


def test_reconstruction_error_shape_and_values():
    X = make_data(60)
    model = MLPAutoencoder(input_dim=8, hidden_dim=16, latent_dim=2, seed=0)
    model.fit(X, epochs=100, lr=0.01)
    errors = model.reconstruction_error(X)
    assert errors.shape == (60,)
    assert np.all(errors >= 0)
    assert np.all(np.isfinite(errors))


def test_anomaly_score_separates_outliers():
    X = make_data(300)
    model = MLPAutoencoder(input_dim=8, hidden_dim=16, latent_dim=2, seed=0)
    model.fit(X, epochs=300, lr=0.01)
    rng = np.random.default_rng(7)
    outliers = X[:30] + rng.normal(0.0, 3.0, size=(30, 8))
    normal = float(np.median(model.anomaly_score(X[:30])))
    outlier = float(np.median(model.anomaly_score(outliers)))
    assert 0.5 < normal < 2.0
    assert outlier > 10 * normal


def test_deterministic_with_seed():
    X = make_data(80)
    a = MLPAutoencoder(input_dim=8, hidden_dim=16, latent_dim=2, seed=3)
    b = MLPAutoencoder(input_dim=8, hidden_dim=16, latent_dim=2, seed=3)
    a.fit(X, epochs=20, lr=0.01)
    b.fit(X, epochs=20, lr=0.01)
    assert a.history_ == b.history_
    assert np.allclose(a.encode(X), b.encode(X))


def test_different_seed_different_weights():
    a = MLPAutoencoder(input_dim=8, hidden_dim=16, latent_dim=2, seed=0)
    b = MLPAutoencoder(input_dim=8, hidden_dim=16, latent_dim=2, seed=1)
    assert not np.allclose(a.W1, b.W1)


def test_single_sample_roundtrip():
    X = make_data(100)
    model = MLPAutoencoder(input_dim=8, hidden_dim=16, latent_dim=2, seed=0)
    model.fit(X, epochs=100, lr=0.01)
    one = X[:1]
    assert model.reconstruct(one).shape == (1, 8)
    assert model.reconstruction_error(one).shape == (1,)


def test_wrong_feature_count_after_fit():
    X = make_data(40)
    model = MLPAutoencoder(input_dim=8, hidden_dim=16, latent_dim=2, seed=0)
    model.fit(X, epochs=10, lr=0.01)
    with pytest.raises(ValueError, match="features"):
        model.encode(np.zeros((2, 5)))
    with pytest.raises(ValueError, match="features"):
        model.decode(np.zeros((2, 5)))
    with pytest.raises(ValueError, match="features"):
        model.reconstruct(np.zeros((2, 5)))


def test_fit_returns_self_and_history_monotone_trend():
    X = make_data(100)
    model = MLPAutoencoder(input_dim=8, hidden_dim=16, latent_dim=2, seed=0)
    assert model.fit(X, epochs=30, lr=0.01) is model
    assert model.is_fitted_
    assert model.train_error_median_ > 0
    # Loss should trend down over training, not necessarily every step.
    assert model.history_[-1] < model.history_[0]
