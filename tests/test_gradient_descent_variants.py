"""Tests for gradient_descent_variants (SGD, Momentum, RMSprop, Adam)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from gradient_descent_variants import (
    SGD,
    Momentum,
    RMSprop,
    Adam,
    minimize,
    compare_optimizers,
    quadratic_loss,
    logistic_loss,
)


def _all_optimizers():
    return [
        SGD(lr=0.1),
        Momentum(lr=0.1, momentum=0.9),
        RMSprop(lr=0.1),
        Adam(lr=0.1),
    ]


# --- single-step behaviour, hand-computed -------------------------------


def test_sgd_step_exact():
    x = np.array([1.0, 2.0])
    g = np.array([0.5, -1.0])
    out = SGD(lr=0.1).step(x, g)
    # x - lr*g = [1 - 0.05, 2 + 0.1]
    np.testing.assert_allclose(out, [0.95, 2.1])


def test_momentum_velocity_accumulates_over_two_steps():
    # Constant grad [1, 1], lr=0.1, mu=0.9, x0=[0, 0]:
    # step 1: v=[1,1], x=[-0.1,-0.1]
    # step 2: v=0.9*[1,1]+[1,1]=[1.9,1.9], x=[-0.1-0.19, -0.1-0.19]
    opt = Momentum(lr=0.1, momentum=0.9)
    x = np.zeros(2)
    g = np.ones(2)
    x = opt.step(x, g)
    np.testing.assert_allclose(x, [-0.1, -0.1])
    x = opt.step(x, g)
    np.testing.assert_allclose(x, [-0.29, -0.29])


def test_rmsprop_first_step_hand_computed():
    # g=[2,-4], decay=0.9: cache = 0.1*g^2 = [0.4, 1.6]
    # step = 0.1*g/(sqrt(cache)+eps) = [0.2/sqrt(0.4), -0.4/sqrt(1.6)]
    #      = [0.31622777, -0.31622777]
    opt = RMSprop(lr=0.1, decay=0.9)
    x = np.array([1.0, 1.0])
    g = np.array([2.0, -4.0])
    out = opt.step(x, g)
    np.testing.assert_allclose(out, [1.0 - 0.31622777, 1.0 + 0.31622777], atol=1e-6)


def test_adam_first_step_approx_lr_times_sign():
    # First step: m_hat = g, v_hat = g^2, so the update is
    # lr * g / (|g| + eps) ~= lr * sign(g).
    opt = Adam(lr=0.01)
    x = np.array([1.0, 1.0])
    g = np.array([3.0, -2.0])
    out = opt.step(x, g)
    np.testing.assert_allclose(out, [0.99, 1.01], atol=1e-6)


# --- step contract ------------------------------------------------------


@pytest.mark.parametrize("opt", _all_optimizers())
def test_step_does_not_mutate_inputs(opt):
    x = np.array([1.0, -2.0, 0.5])
    g = np.array([0.3, -0.7, 1.2])
    x_before, g_before = x.copy(), g.copy()
    opt.step(x, g)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(g, g_before)


@pytest.mark.parametrize("opt", _all_optimizers())
def test_step_returns_new_array(opt):
    x = np.array([1.0, 2.0])
    out = opt.step(x, np.array([0.5, 0.5]))
    assert out is not x
    assert out.shape == x.shape


def test_separate_instances_have_independent_state():
    # Two momentum optimizers stepped a different number of times must
    # not share velocity; the less-stepped one must behave like a fresh one.
    opt1 = Momentum(lr=0.1, momentum=0.9)
    opt2 = Momentum(lr=0.1, momentum=0.9)
    fresh = Momentum(lr=0.1, momentum=0.9)
    x, g = np.zeros(2), np.ones(2)
    opt1.step(x, g)
    opt1.step(x, g)  # opt1 now has velocity 1.9
    out2 = opt2.step(x, g)
    out_fresh = fresh.step(x, g)
    np.testing.assert_allclose(out2, out_fresh)
    np.testing.assert_allclose(out2, [-0.1, -0.1])


# --- minimize / compare_optimizers --------------------------------------


def _simple_quadratic():
    A = np.diag([1.0, 2.0])
    b = np.array([1.0, 1.0])
    return quadratic_loss(A, b), np.linalg.solve(A, b)


def test_minimize_reduces_loss_on_quadratic():
    objective, _ = _simple_quadratic()
    x_final, history = minimize(objective, np.zeros(2), SGD(lr=0.1), n_steps=200)
    assert len(history) == 200
    assert history[-1] < history[0]
    assert history[-1] < 0  # loss at x0=0 is 0; optimum loss is -0.75


def test_minimize_converges_close_to_optimum():
    objective, x_star = _simple_quadratic()
    x_final, history = minimize(objective, np.zeros(2), SGD(lr=0.1), n_steps=200)
    np.testing.assert_allclose(x_final, x_star, atol=1e-3)
    # Optimum loss: 0.5*(1 + 2*0.25) - 1.5 = -0.75
    assert abs(history[-1] - (-0.75)) < 1e-6


def test_adam_converges_on_quadratic():
    objective, x_star = _simple_quadratic()
    x_final, history = minimize(objective, np.zeros(2), Adam(lr=0.05), n_steps=500)
    assert history[-1] < history[0]
    assert history[-1] < -0.70  # optimum is -0.75; Adam oscillates a little
    assert np.linalg.norm(x_final - x_star) < 0.1


def test_compare_optimizers_returns_all_four():
    objective, _ = _simple_quadratic()
    results = compare_optimizers(objective, np.zeros(2), n_steps=100)
    assert set(results) == {"SGD", "Momentum", "RMSprop", "Adam"}
    for name, res in results.items():
        assert len(res["history"]) == 100
        assert res["x"].shape == (2,)
        assert isinstance(res["loss"], float)
        assert res["history"][-1] < res["history"][0]


def test_compare_optimizers_accepts_custom_optimizers():
    objective, _ = _simple_quadratic()
    results = compare_optimizers(
        objective, np.zeros(2), n_steps=10,
        optimizers={"slow": SGD(lr=0.001), "fast": SGD(lr=0.1)},
    )
    assert set(results) == {"slow", "fast"}
    assert results["fast"]["loss"] < results["slow"]["loss"]


# --- objectives ----------------------------------------------------------


def test_quadratic_loss_optimum_is_a_inverse_b():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 2.0])
    objective = quadratic_loss(A, b)
    x_star = np.linalg.solve(A, b)
    loss_star, grad_star = objective(x_star)
    np.testing.assert_allclose(grad_star, [0.0, 0.0], atol=1e-10)
    loss_off, _ = objective(x_star + np.array([0.1, -0.2]))
    assert loss_star < loss_off


def test_logistic_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 2))
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    objective = logistic_loss(X, y)
    w = np.array([0.3, -0.7])
    _, grad = objective(w)
    eps = 1e-6
    numeric = np.zeros_like(w)
    for i in range(len(w)):
        dw = np.zeros_like(w)
        dw[i] = eps
        numeric[i] = (objective(w + dw)[0] - objective(w - dw)[0]) / (2 * eps)
    np.testing.assert_allclose(grad, numeric, atol=1e-5)


def test_minimize_reduces_logistic_loss():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(float)
    objective = logistic_loss(X, y)
    _, history = minimize(objective, np.zeros(2), Adam(lr=0.05), n_steps=200)
    assert history[-1] < history[0]
    assert history[-1] < 0.3


# --- validation ----------------------------------------------------------


@pytest.mark.parametrize("bad_lr", [0.0, -0.01])
def test_invalid_lr_raises(bad_lr):
    with pytest.raises(ValueError):
        SGD(lr=bad_lr)
    with pytest.raises(ValueError):
        Adam(lr=bad_lr)


@pytest.mark.parametrize("bad_momentum", [-0.1, 1.0, 1.5])
def test_invalid_momentum_raises(bad_momentum):
    with pytest.raises(ValueError):
        Momentum(lr=0.01, momentum=bad_momentum)


@pytest.mark.parametrize("bad_rate", [0.0, 1.0, -0.5, 2.0])
def test_invalid_decay_beta_raises(bad_rate):
    with pytest.raises(ValueError):
        RMSprop(lr=0.01, decay=bad_rate)
    with pytest.raises(ValueError):
        Adam(lr=0.001, beta1=bad_rate)
    with pytest.raises(ValueError):
        Adam(lr=0.001, beta2=bad_rate)


@pytest.mark.parametrize("bad_eps", [0.0, -1e-8])
def test_invalid_eps_raises(bad_eps):
    with pytest.raises(ValueError):
        RMSprop(lr=0.01, eps=bad_eps)
    with pytest.raises(ValueError):
        Adam(lr=0.001, eps=bad_eps)


def test_minimize_invalid_n_steps():
    objective, _ = _simple_quadratic()
    with pytest.raises(ValueError):
        minimize(objective, np.zeros(2), SGD(), n_steps=0)


def test_minimize_empty_x0():
    objective, _ = _simple_quadratic()
    with pytest.raises(ValueError):
        minimize(objective, np.array([]), SGD(), n_steps=10)


def test_minimize_bad_objective_return():
    def bad_tuple(x):
        return 1.0  # not a (loss, grad) tuple

    def bad_shape(x):
        return 1.0, np.array([1.0, 2.0, 3.0])  # wrong grad shape

    with pytest.raises(ValueError):
        minimize(bad_tuple, np.zeros(2), SGD(), n_steps=5)
    with pytest.raises(ValueError):
        minimize(bad_shape, np.zeros(2), SGD(), n_steps=5)


def test_logistic_nonbinary_y_raises():
    X = np.ones((3, 2))
    with pytest.raises(ValueError):
        logistic_loss(X, np.array([0.0, 1.0, 2.0]))


def test_quadratic_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        quadratic_loss(np.eye(2), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        quadratic_loss(np.ones((2, 3)), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        quadratic_loss(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 1.0]))


def test_step_shape_mismatch_raises():
    with pytest.raises(ValueError):
        SGD().step(np.zeros(2), np.zeros(3))
