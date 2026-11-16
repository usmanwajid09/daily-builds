import numpy as np
import pytest

from digit_recognizer import (
    TEMPLATES,
    add_noise,
    accuracy,
    distances,
    recognize,
    template,
)


def test_all_templates_have_correct_shape_and_values():
    assert len(TEMPLATES) == 10
    for digit in range(10):
        tmpl = template(digit)
        assert tmpl.shape == (8, 8)
        assert set(np.unique(tmpl)).issubset({0.0, 1.0})


def test_exact_templates_recognised():
    for digit in range(10):
        pred, scores = recognize(template(digit))
        assert pred == digit
        assert scores[digit] == pytest.approx(0.0)


def test_noisy_variants_recognised_with_high_accuracy():
    rng = np.random.default_rng(0)
    images, labels = [], []
    for digit in range(10):
        for _ in range(20):
            images.append(add_noise(template(digit), 0.05, rng))
            labels.append(digit)
    assert accuracy(images, labels) >= 0.95


def test_distance_to_own_template_is_smallest():
    scores = distances(template(4))
    assert min(scores, key=scores.get) == 4
    assert scores[4] == pytest.approx(0.0)
    assert all(scores[d] > 0.0 for d in scores if d != 4)


def test_add_noise_with_zero_probability_is_identity():
    clean = template(2)
    assert np.array_equal(add_noise(clean, 0.0), clean)


def test_add_noise_with_probability_one_inverts():
    clean = template(2)
    assert np.array_equal(add_noise(clean, 1.0), 1.0 - clean)


def test_invalid_digit_raises():
    with pytest.raises(ValueError):
        template(10)
    with pytest.raises(ValueError):
        template(-1)


def test_invalid_image_shape_raises():
    with pytest.raises(ValueError):
        recognize(np.zeros((7, 8)))
    with pytest.raises(ValueError):
        recognize(np.zeros((8, 8, 1)))


def test_out_of_range_values_raise():
    with pytest.raises(ValueError):
        recognize(np.full((8, 8), 2.0))


def test_invalid_flip_probability_raises():
    with pytest.raises(ValueError):
        add_noise(template(0), 1.5)


def test_accuracy_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        accuracy([template(0)], [0, 1])


def test_accuracy_empty_raises():
    with pytest.raises(ValueError):
        accuracy([], [])
