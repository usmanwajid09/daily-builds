"""Tests for cnn_from_scratch."""
import numpy as np
import pytest

from cnn_from_scratch.cnn import (
    Conv2D,
    Dense,
    MaxPool2D,
    ReLU,
    TinyCNN,
    cross_entropy_loss,
    make_circle,
    make_shape_dataset,
    make_square,
    softmax,
)


# --- softmax / loss -------------------------------------------------------
def test_softmax_sums_to_one():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert p.shape == (3,)
    assert abs(p.sum() - 1.0) < 1e-9
    assert np.all(p > 0)


def test_softmax_stable_for_large_logits():
    p = softmax(np.array([1000.0, 1001.0]))
    assert np.all(np.isfinite(p))
    assert abs(p.sum() - 1.0) < 1e-9


def test_softmax_rejects_scalar():
    with pytest.raises(ValueError):
        softmax(1.0)


def test_cross_entropy_loss_known_value():
    loss = cross_entropy_loss(np.array([[0.5, 0.5]]), np.array([0]))
    assert abs(loss - np.log(2)) < 1e-9


def test_cross_entropy_rejects_bad_label():
    with pytest.raises(ValueError):
        cross_entropy_loss(np.array([[0.5, 0.5]]), np.array([2]))


def test_cross_entropy_rejects_mismatched_labels():
    with pytest.raises(ValueError):
        cross_entropy_loss(np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([0]))


# --- layers ---------------------------------------------------------------
def test_conv_output_shape():
    conv = Conv2D(n_filters=3, kernel_size=3, rng=0)
    out = conv.forward(np.zeros((5, 5)))
    assert out.shape == (3, 3, 3)


def test_conv_rejects_too_small_input():
    conv = Conv2D(n_filters=2, kernel_size=3, rng=0)
    with pytest.raises(ValueError):
        conv.forward(np.zeros((2, 2)))


def test_conv_rejects_bad_params():
    with pytest.raises(ValueError):
        Conv2D(n_filters=0)
    with pytest.raises(ValueError):
        Conv2D(n_filters=2, kernel_size=0)


def test_conv_backward_shapes():
    conv = Conv2D(n_filters=2, kernel_size=3, rng=0)
    conv.forward(np.random.default_rng(0).normal(size=(5, 5)))
    dx = conv.backward(np.ones((3, 3, 2)))
    assert dx.shape == (5, 5)
    assert conv.d_filters.shape == (2, 3, 3)


def test_relu_forward_backward():
    relu = ReLU()
    out = relu.forward(np.array([-1.0, 0.0, 2.0]))
    assert out.tolist() == [0.0, 0.0, 2.0]
    grad = relu.backward(np.array([1.0, 1.0, 1.0]))
    assert grad.tolist() == [0.0, 0.0, 1.0]


def test_maxpool_forward_values():
    pool = MaxPool2D(2)
    x = np.arange(16, dtype=float).reshape(4, 4, 1)
    out = pool.forward(x)
    assert out.shape == (2, 2, 1)
    assert out[0, 0, 0] == 5.0  # max of [[0,1],[4,5]]
    assert out[1, 1, 0] == 15.0


def test_maxpool_backward_routes_to_argmax():
    pool = MaxPool2D(2)
    x = np.arange(16, dtype=float).reshape(4, 4, 1)
    pool.forward(x)
    dx = pool.backward(np.ones((2, 2, 1)))
    assert dx.shape == (4, 4, 1)
    assert dx.sum() == 4.0  # one unit of gradient per pooled cell
    assert dx[1, 1, 0] == 1.0  # argmax of the top-left window


def test_dense_forward_shape_and_validation():
    dense = Dense(4, 3, rng=0)
    assert dense.forward(np.ones(4)).shape == (3,)
    with pytest.raises(ValueError):
        dense.forward(np.ones(5))
    with pytest.raises(ValueError):
        Dense(0, 3)


# --- network --------------------------------------------------------------
def test_tinycnn_forward_probs_sum_to_one():
    net = TinyCNN(input_shape=(12, 12), n_filters=4, n_classes=2, seed=0)
    probs = net.forward(np.zeros((12, 12)))
    assert probs.shape == (2,)
    assert abs(probs.sum() - 1.0) < 1e-9


def test_tinycnn_rejects_wrong_input_shape():
    net = TinyCNN(input_shape=(12, 12), seed=0)
    with pytest.raises(ValueError):
        net.forward(np.zeros((10, 10)))


def test_tinycnn_rejects_bad_n_classes():
    with pytest.raises(ValueError):
        TinyCNN(n_classes=1)


def test_tinycnn_fit_reduces_loss():
    X, y = make_shape_dataset(n_per_class=20, size=12, seed=1)
    net = TinyCNN(input_shape=(12, 12), n_filters=4, n_classes=2, seed=0)
    losses = net.fit(X, y, epochs=10, lr=0.05)
    assert len(losses) == 10
    assert losses[-1] < losses[0]


def test_tinycnn_learns_shapes():
    X_train, y_train = make_shape_dataset(n_per_class=40, size=12, seed=0)
    X_test, y_test = make_shape_dataset(n_per_class=10, size=12, seed=99)
    net = TinyCNN(input_shape=(12, 12), n_filters=4, n_classes=2, seed=0)
    net.fit(X_train, y_train, epochs=30, lr=0.05)
    assert net.score(X_test, y_test) >= 0.8


def test_tinycnn_fit_validates_labels():
    X, y = make_shape_dataset(n_per_class=5, size=12, seed=0)
    net = TinyCNN(input_shape=(12, 12), seed=0)
    with pytest.raises(ValueError):
        net.fit(X, np.array([0] * 9 + [5]), epochs=1)
    with pytest.raises(ValueError):
        net.fit(X, y, epochs=0)
    with pytest.raises(ValueError):
        net.fit(X, y, epochs=1, lr=-0.1)


def test_predict_returns_valid_labels():
    X, _ = make_shape_dataset(n_per_class=5, size=12, seed=0)
    net = TinyCNN(input_shape=(12, 12), seed=0)
    preds = net.predict(X)
    assert preds.shape == (10,)
    assert set(preds.tolist()) <= {0, 1}


# --- data -----------------------------------------------------------------
def test_shape_dataset_shapes_and_balance():
    X, y = make_shape_dataset(n_per_class=10, size=12, seed=0)
    assert X.shape == (20, 12, 12)
    assert y.shape == (20,)
    assert int((y == 0).sum()) == 10
    assert int((y == 1).sum()) == 10
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_make_circle_and_square_differ():
    rng = np.random.default_rng(0)
    c = make_circle(size=12, centre=(6, 6), radius=3, rng=rng)
    s = make_square(size=12, centre=(6, 6), half_side=3, rng=rng)
    assert c.shape == s.shape == (12, 12)
    assert not np.array_equal(c, s)
    assert c.sum() > 0 and s.sum() > 0


def test_make_circle_rejects_bad_radius():
    with pytest.raises(ValueError):
        make_circle(size=12, radius=0)
