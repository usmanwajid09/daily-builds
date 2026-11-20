"""Tests for histogram_matching."""

import numpy as np
import pytest

from histogram_matching import (
    compute_cdf,
    compute_histogram,
    match_histogram,
)
from histogram_matching.demo import make_images


def test_match_to_itself_is_identity():
    source, _ = make_images()
    matched = match_histogram(source, source)
    assert np.array_equal(matched, source)


def test_matched_stats_close_to_reference():
    source, reference = make_images()
    matched = match_histogram(source, reference)
    assert matched.dtype == np.uint8
    assert matched.shape == source.shape
    assert abs(matched.mean() - reference.mean()) < 2.0
    assert abs(matched.std() - reference.std()) < 2.0


def test_matching_is_monotonic():
    # A monotone LUT: brighter source pixels stay brighter.
    source, reference = make_images()
    matched = match_histogram(source, reference)
    order = np.argsort(source.ravel())
    assert np.all(np.diff(matched.ravel()[order]) >= 0)


def test_constant_source_stays_constant():
    source = np.full((10, 12), 77, dtype=np.uint8)
    _, reference = make_images()
    matched = match_histogram(source, reference)
    assert np.all(matched == matched.flat[0])


def test_rgb_per_channel():
    rng = np.random.default_rng(3)
    source = rng.integers(0, 100, size=(16, 16, 3), dtype=np.uint8)
    reference = rng.integers(150, 256, size=(16, 16, 3), dtype=np.uint8)
    matched = match_histogram(source, reference)
    assert matched.shape == source.shape
    assert matched.dtype == np.uint8
    for c in range(3):
        assert abs(matched[:, :, c].mean() - reference[:, :, c].mean()) < 3.0


def test_float01_round_trip():
    source, reference = make_images()
    src_f = source.astype(np.float64) / 255.0
    ref_f = reference.astype(np.float64) / 255.0
    matched = match_histogram(src_f, ref_f)
    assert matched.dtype == np.float64
    assert matched.min() >= 0.0 and matched.max() <= 1.0
    assert abs(matched.mean() - ref_f.mean()) < 0.01


def test_float255_input():
    source = np.full((8, 8), 50.0)
    reference = np.full((8, 8), 200.0)
    matched = match_histogram(source, reference)
    assert matched.dtype == np.float64
    assert np.all(matched == 200.0)


def test_mismatched_kinds_raise():
    source, reference = make_images()
    rgb = np.stack([source] * 3, axis=-1)
    with pytest.raises(ValueError):
        match_histogram(source, rgb)
    with pytest.raises(ValueError):
        match_histogram(rgb, reference)


def test_invalid_inputs_raise():
    source, reference = make_images()
    with pytest.raises(ValueError):
        match_histogram(source.ravel(), reference)
    with pytest.raises(ValueError):
        match_histogram(np.zeros((0, 0), dtype=np.uint8), reference)
    with pytest.raises(ValueError):
        match_histogram(source, reference, bins=1)
    with pytest.raises(ValueError):
        match_histogram(source.astype(np.float64) * 300.0, reference)


def test_compute_histogram_counts():
    source, _ = make_images()
    hist = compute_histogram(source)
    assert hist.shape == (256,)
    assert hist.sum() == source.size
    assert hist.dtype == np.int64
    # The source is dark: no mass above level 112.
    assert hist[113:].sum() == 0


def test_compute_histogram_rejects_rgb():
    source, _ = make_images()
    rgb = np.stack([source] * 3, axis=-1)
    with pytest.raises(ValueError):
        compute_histogram(rgb)


def test_compute_cdf_properties():
    source, _ = make_images()
    cdf = compute_cdf(source)
    assert cdf.shape == (256,)
    assert np.all(np.diff(cdf) >= 0)
    assert cdf[-1] == pytest.approx(1.0)
    assert cdf[0] >= 0.0
