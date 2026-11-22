import numpy as np
import pytest

from active_contour_snake.active_contour_snake import (
    ActiveContour,
    _mean_radius,
    contour_length,
    enclosed_area,
)
from active_contour_snake.demo import make_disc_image

CENTER = (32.0, 32.0)
TRUE_RADIUS = 15.0
INIT_RADIUS = 25.0


def _snake(**kwargs):
    params = dict(alpha=0.5, beta=0.2, gamma=1.0,
                  n_iterations=100, n_points=50)
    params.update(kwargs)
    return ActiveContour(**params)


def _fitted():
    """Fit the default snake to the default disc; return (snake, image, final)."""
    image = make_disc_image(64, 64, CENTER, TRUE_RADIUS)
    snake = _snake()
    initial = snake.initialise_circle(CENTER, INIT_RADIUS,
                                      image_shape=image.shape)
    return snake, image, snake.fit(image, initial)


# -- convergence and energy -----------------------------------------------

def test_snake_converges_to_true_circle_radius():
    snake, image, final = _fitted()
    assert abs(_mean_radius(final, CENTER) - TRUE_RADIUS) < 3.0


def test_energy_decreases_after_fitting():
    image = make_disc_image(64, 64, CENTER, TRUE_RADIUS)
    snake = _snake()
    initial = snake.initialise_circle(CENTER, INIT_RADIUS,
                                      image_shape=image.shape)
    assert snake.energy(image, snake.fit(image, initial)) \
        < snake.energy(image, initial)


def test_contour_stays_inside_image_bounds():
    _, image, final = _fitted()
    assert np.all(final[:, 0] >= 0) and np.all(final[:, 0] < image.shape[1])
    assert np.all(final[:, 1] >= 0) and np.all(final[:, 1] < image.shape[0])


def test_fit_is_deterministic():
    image = make_disc_image(64, 64, CENTER, TRUE_RADIUS)
    snake = _snake()
    initial = snake.initialise_circle(CENTER, INIT_RADIUS,
                                      image_shape=image.shape)
    first = snake.fit(image, initial)
    second = snake.fit(image, initial)
    np.testing.assert_array_equal(first, second)


def test_fit_blank_image_does_not_crash():
    image = np.zeros((64, 64), dtype=np.float64)
    snake = _snake()
    initial = snake.initialise_circle(CENTER, INIT_RADIUS,
                                      image_shape=image.shape)
    final = snake.fit(image, initial)
    assert np.all(np.isfinite(final))
    assert np.all(final[:, 0] >= 0) and np.all(final[:, 0] < 64)
    assert np.all(final[:, 1] >= 0) and np.all(final[:, 1] < 64)


def test_fit_does_not_modify_input_contour():
    image = make_disc_image(64, 64, CENTER, TRUE_RADIUS)
    snake = _snake()
    initial = snake.initialise_circle(CENTER, INIT_RADIUS,
                                      image_shape=image.shape)
    before = initial.copy()
    snake.fit(image, initial)
    np.testing.assert_array_equal(initial, before)


# -- geometry helpers ------------------------------------------------------

def test_contour_length_of_circle():
    snake = _snake()
    circle = snake.initialise_circle(CENTER, 10.0)
    assert contour_length(circle) == pytest.approx(2 * np.pi * 10.0, rel=0.01)


def test_enclosed_area_of_circle():
    snake = _snake()
    circle = snake.initialise_circle(CENTER, 10.0)
    assert enclosed_area(circle) == pytest.approx(np.pi * 10.0 ** 2, rel=0.01)


def test_initialise_circle_radius_and_center():
    snake = _snake(n_points=100)
    circle = snake.initialise_circle(CENTER, 12.0, image_shape=(64, 64))
    assert circle.shape == (100, 2)
    assert _mean_radius(circle, CENTER) == pytest.approx(12.0, abs=1e-9)
    assert np.mean(circle, axis=0) == pytest.approx(CENTER, abs=1e-9)


# -- invalid inputs ---------------------------------------------------------

def test_invalid_1d_image():
    snake = _snake()
    with pytest.raises(ValueError):
        snake.fit(np.zeros(64), snake.initialise_circle(CENTER, 10.0))


def test_invalid_empty_image():
    snake = _snake()
    with pytest.raises(ValueError):
        snake.fit(np.zeros((0, 0)), snake.initialise_circle(CENTER, 10.0))


def test_invalid_nan_image():
    image = make_disc_image(64, 64, CENTER, TRUE_RADIUS)
    image[10, 10] = np.nan
    snake = _snake()
    with pytest.raises(ValueError):
        snake.fit(image, snake.initialise_circle(CENTER, INIT_RADIUS))


def test_contour_with_fewer_than_three_points():
    snake = _snake()
    image = make_disc_image(64, 64, CENTER, TRUE_RADIUS)
    with pytest.raises(ValueError):
        snake.fit(image, np.array([[10.0, 10.0], [20.0, 20.0]]))


def test_contour_with_wrong_shape():
    snake = _snake()
    image = make_disc_image(64, 64, CENTER, TRUE_RADIUS)
    with pytest.raises(ValueError):
        snake.fit(image, np.zeros((5, 3)))


@pytest.mark.parametrize("radius", [0.0, -5.0])
def test_radius_must_be_positive(radius):
    snake = _snake()
    with pytest.raises(ValueError):
        snake.initialise_circle(CENTER, radius, image_shape=(64, 64))


def test_center_outside_image():
    snake = _snake()
    with pytest.raises(ValueError):
        snake.initialise_circle((100.0, 100.0), 10.0, image_shape=(64, 64))


@pytest.mark.parametrize("kwargs", [
    {"alpha": -0.1},
    {"beta": -0.1},
])
def test_negative_alpha_or_beta_rejected(kwargs):
    with pytest.raises(ValueError):
        ActiveContour(**kwargs)


def test_gamma_must_be_positive():
    with pytest.raises(ValueError):
        ActiveContour(gamma=0.0)


def test_n_iterations_must_be_at_least_one():
    with pytest.raises(ValueError):
        ActiveContour(n_iterations=0)
