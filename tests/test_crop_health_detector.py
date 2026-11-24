import numpy as np
import pytest

from crop_health_detector.crop_health import (
    BARE,
    HEALTHY,
    STRESSED,
    classification_accuracy,
    classify_patch,
    evi,
    health_map,
    make_field,
    ndvi,
    zone_mean_ndvi,
)


# ---------------------------------------------------------------- make_field

def test_make_field_shapes_and_ranges():
    red, nir, labels = make_field(64, 64, seed=42)
    assert red.shape == (64, 64) and nir.shape == (64, 64)
    assert labels.shape == (64, 64)
    assert red.min() >= 0.0 and red.max() <= 1.0
    assert nir.min() >= 0.0 and nir.max() <= 1.0
    assert set(np.unique(labels)) == {HEALTHY, STRESSED, BARE}


def test_make_field_deterministic():
    r1, n1, l1 = make_field(32, 32, seed=7)
    r2, n2, l2 = make_field(32, 32, seed=7)
    assert np.array_equal(r1, r2) and np.array_equal(n1, n2)
    assert np.array_equal(l1, l2)
    r3, _, _ = make_field(32, 32, seed=8)
    assert not np.array_equal(r1, r3)


def test_make_field_zone_signatures():
    red, nir, labels = make_field(64, 64, seed=42)
    ndvi_map = ndvi(red, nir)
    means = zone_mean_ndvi(ndvi_map, labels)
    assert 0.65 < means["healthy"] < 0.90
    assert 0.20 < means["stressed"] < 0.50
    assert -0.10 < means["bare"] < 0.15
    assert means["healthy"] > means["stressed"] > means["bare"]


def test_make_field_invalid_sizes():
    with pytest.raises(ValueError):
        make_field(0, 64)
    with pytest.raises(ValueError):
        make_field(64, -1)
    with pytest.raises(ValueError):
        make_field(64.5, 64)


def test_make_field_odd_size_still_works():
    red, nir, labels = make_field(5, 7, seed=1)
    assert red.shape == (5, 7)
    assert set(np.unique(labels)) == {HEALTHY, STRESSED, BARE}


# ---------------------------------------------------------------------- ndvi

def test_ndvi_healthy_vegetation_high():
    assert ndvi(np.array([[0.10]]), np.array([[0.80]]))[0, 0] == pytest.approx(0.778, abs=1e-3)


def test_ndvi_bare_soil_near_zero():
    assert abs(ndvi(np.array([[0.30]]), np.array([[0.32]]))[0, 0]) < 0.05


def test_ndvi_zero_denominator_is_zero():
    assert ndvi(np.array([[0.0]]), np.array([[0.0]]))[0, 0] == 0.0


def test_ndvi_mismatched_shapes():
    with pytest.raises(ValueError, match="mismatched shapes"):
        ndvi(np.zeros((2, 2)), np.zeros((3, 3)))


def test_ndvi_negative_reflectance():
    with pytest.raises(ValueError, match="negative"):
        ndvi(np.array([[-0.1]]), np.array([[0.5]]))


def test_ndvi_reflectance_above_one():
    with pytest.raises(ValueError, match="above 1"):
        ndvi(np.array([[0.5]]), np.array([[1.2]]))


def test_ndvi_empty_array():
    with pytest.raises(ValueError, match="empty"):
        ndvi(np.array([]), np.array([]))


# ----------------------------------------------------------------------- evi

def test_evi2_healthy_above_stressed():
    healthy = evi(np.array([[0.80]]), np.array([[0.10]]))[0, 0]
    stressed = evi(np.array([[0.50]]), np.array([[0.25]]))[0, 0]
    bare = evi(np.array([[0.30]]), np.array([[0.30]]))[0, 0]
    assert healthy > stressed > bare
    assert healthy == pytest.approx(0.858, abs=0.02)


def test_evi_with_blue_band_uses_three_band_formula():
    # 2.5*(0.8-0.1)/(0.8+0.6-0.75+1) = 1.75/1.65
    got = evi(np.array([[0.80]]), np.array([[0.10]]), blue=np.array([[0.10]]))[0, 0]
    assert got == pytest.approx(1.75 / 1.65, abs=1e-6)


def test_evi_blue_shape_mismatch():
    with pytest.raises(ValueError, match="mismatched shapes"):
        evi(np.zeros((2, 2)), np.zeros((2, 2)), blue=np.zeros((2, 3)))


# ------------------------------------------------------------ classify_patch

def test_classify_patch_each_class():
    assert classify_patch(0.80, 0.70) == "healthy"
    assert classify_patch(0.35, 0.30) == "stressed"
    assert classify_patch(0.05, 0.02) == "bare"


def test_classify_patch_array_input_uses_mean():
    assert classify_patch(np.array([0.8, 0.75]), np.array([0.7, 0.65])) == "healthy"
    assert classify_patch(np.array([0.05, 0.10]), np.array([0.0, 0.05])) == "bare"


def test_classify_patch_bad_thresholds():
    with pytest.raises(ValueError, match="thresholds"):
        classify_patch(0.8, 0.7, healthy_ndvi=0.2, stressed_ndvi=0.5)


def test_classify_patch_ndvi_out_of_range():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        classify_patch(1.5, 0.7)


# --------------------------------------------------------------- health_map

def test_health_map_counts_and_percentages():
    ndvi_map = np.array([[0.8, 0.8], [0.3, 0.05]])
    labels, summary = health_map(ndvi_map)
    assert labels.tolist() == [[HEALTHY, HEALTHY], [STRESSED, BARE]]
    assert summary["total_pixels"] == 4
    assert summary["counts"] == {"healthy": 2, "stressed": 1, "bare": 1}
    assert summary["percentages"]["healthy"] == pytest.approx(50.0)
    assert sum(summary["percentages"].values()) == pytest.approx(100.0)


def test_health_map_with_evi_refinement():
    # NDVI says healthy but EVI is too low -> falls back to stressed/bare.
    ndvi_map = np.array([[0.8]])
    labels, _ = health_map(ndvi_map, evi_values=np.array([[0.05]]))
    assert labels[0, 0] == BARE  # below even the stressed EVI threshold


def test_health_map_end_to_end_accuracy():
    red, nir, truth = make_field(64, 64, seed=42)
    labels, summary = health_map(ndvi(red, nir), evi(nir, red))
    assert classification_accuracy(labels, truth) >= 0.90


def test_health_map_empty_and_bad_ndvi():
    with pytest.raises(ValueError, match="empty"):
        health_map(np.array([]))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        health_map(np.array([[1.2]]))


# ------------------------------------------------- zone stats and accuracy

def test_zone_mean_ndvi_values():
    ndvi_map = np.array([[0.8, 0.3], [0.05, 0.7]])
    labels = np.array([[HEALTHY, STRESSED], [BARE, HEALTHY]])
    means = zone_mean_ndvi(ndvi_map, labels)
    assert means["healthy"] == pytest.approx(0.75)
    assert means["stressed"] == pytest.approx(0.3)
    assert means["bare"] == pytest.approx(0.05)


def test_zone_mean_ndvi_invalid_label():
    with pytest.raises(ValueError, match="labels must contain only"):
        zone_mean_ndvi(np.array([[0.5]]), np.array([[5]]))


def test_classification_accuracy_perfect_and_half():
    truth = np.array([[HEALTHY, STRESSED], [BARE, HEALTHY]])
    assert classification_accuracy(truth, truth) == 1.0
    pred = np.array([[HEALTHY, BARE], [BARE, STRESSED]])
    assert classification_accuracy(pred, truth) == pytest.approx(0.5)


def test_classification_accuracy_shape_mismatch():
    with pytest.raises(ValueError, match="mismatched shapes"):
        classification_accuracy(np.zeros((2, 2), dtype=int), np.zeros((2, 3), dtype=int))
