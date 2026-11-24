"""Tests for signature_verifier.verifier (synthetic signature verification)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from signature_verifier.verifier import (
    DEFAULT_THRESHOLD,
    SignatureVerifier,
    create_enrollment,
    dtw_distance,
    enroll,
    extract_features,
    make_signature,
    normalise_trajectory,
    verify,
)

N = 60  # shorter trajectories keep the pure-Python DTW fast in tests


def _enrolled(person="usman", n_refs=2):
    verifier = SignatureVerifier()
    verifier.enroll(person, [make_signature(person, seed=100 + i, n_points=N) for i in range(n_refs)])
    return verifier


# --- make_signature -------------------------------------------------------

def test_make_signature_shape_and_determinism():
    sig = make_signature("usman", seed=1, n_points=N)
    assert sig.shape == (N, 2)
    assert np.all(np.isfinite(sig))
    again = make_signature("usman", seed=1, n_points=N)
    assert np.array_equal(sig, again)


def test_make_signature_different_people_differ():
    a = make_signature("usman", seed=1, n_points=N)
    b = make_signature("alice", seed=1, n_points=N)
    assert not np.allclose(a, b)


def test_make_signature_different_seeds_differ():
    a = make_signature("usman", seed=1, n_points=N)
    b = make_signature("usman", seed=2, n_points=N)
    assert not np.allclose(a, b)


def test_make_signature_forgery_differs_from_genuine():
    genuine = make_signature("usman", genuine=True, seed=1, n_points=N)
    forgery = make_signature("usman", genuine=False, seed=1, n_points=N)
    assert not np.allclose(genuine, forgery)


def test_make_signature_invalid_person_and_n_points():
    with pytest.raises(ValueError, match="person_id"):
        make_signature("", seed=1)
    with pytest.raises(ValueError, match="person_id"):
        make_signature(123, seed=1)
    with pytest.raises(ValueError, match="n_points"):
        make_signature("usman", seed=1, n_points=5)
    with pytest.raises(ValueError, match="seed"):
        make_signature("usman", seed="x")


# --- extract_features -----------------------------------------------------

def test_extract_features_keys_and_ranges():
    feats = extract_features(make_signature("usman", seed=1, n_points=N))
    assert feats["n_points"] == N
    assert feats["aspect_ratio"] > 0
    assert feats["total_path_length"] > 0
    assert feats["mean_curvature"] >= 0
    assert feats["std_curvature"] >= 0
    assert feats["mean_speed"] > 0
    assert feats["std_speed"] >= 0
    assert feats["n_direction_changes"] >= 0
    assert feats["bbox_fill"] > 0


def test_extract_features_forgery_has_more_curvature():
    genuine = extract_features(make_signature("usman", genuine=True, seed=1, n_points=N))
    forgery = extract_features(make_signature("usman", genuine=False, seed=1, n_points=N))
    # Extra wobble in forgeries bends the stroke more.
    assert forgery["mean_curvature"] > genuine["mean_curvature"]


def test_extract_features_with_timestamps():
    sig = make_signature("usman", seed=1, n_points=N)
    times = np.linspace(0.0, 2.0, N)
    feats = extract_features(sig, timestamps=times)
    assert feats["mean_speed"] > 0
    assert feats["n_points"] == N


def test_extract_features_invalid_input():
    with pytest.raises(ValueError, match="too few points"):
        extract_features(np.zeros((5, 2)))
    with pytest.raises(ValueError, match="2D"):
        extract_features(np.zeros((N, 3)))
    with pytest.raises(ValueError, match="2D"):
        extract_features(np.zeros(N))
    with pytest.raises(ValueError, match="timestamps"):
        extract_features(make_signature("usman", seed=1, n_points=N), timestamps=[0.0, 1.0])
    with pytest.raises(ValueError, match="timestamps"):
        extract_features(
            make_signature("usman", seed=1, n_points=N),
            timestamps=np.linspace(0.0, 1.0, N)[::-1],  # decreasing
        )


# --- dtw_distance ---------------------------------------------------------

def test_dtw_distance_identical_is_zero():
    sig = make_signature("usman", seed=1, n_points=N)
    assert dtw_distance(sig, sig) == pytest.approx(0.0, abs=1e-9)


def test_dtw_distance_symmetric_and_genuine_closer_than_forgery():
    ref = make_signature("usman", seed=1, n_points=N)
    genuine = make_signature("usman", seed=2, n_points=N)
    forgery = make_signature("usman", genuine=False, seed=2, n_points=N)
    assert dtw_distance(ref, genuine) == pytest.approx(dtw_distance(genuine, ref), rel=1e-6)
    assert dtw_distance(ref, genuine) < dtw_distance(ref, forgery)


def test_dtw_distance_translation_and_scale_invariant():
    sig = make_signature("usman", seed=1, n_points=N)
    moved = sig * 3.0 + np.array([100.0, -50.0])
    assert dtw_distance(sig, moved) == pytest.approx(0.0, abs=1e-9)


def test_dtw_distance_invalid_input():
    sig = make_signature("usman", seed=1, n_points=N)
    with pytest.raises(ValueError, match="too few points"):
        dtw_distance(np.zeros((1, 2)), sig)
    with pytest.raises(ValueError, match="2D"):
        dtw_distance(np.zeros((N, 3)), sig)
    with pytest.raises(ValueError, match="window"):
        dtw_distance(sig, sig, window=0)


def test_normalise_trajectory_centres_and_scales():
    sig = make_signature("usman", seed=1, n_points=N) * 5.0 + 10.0
    normed = normalise_trajectory(sig)
    assert np.allclose(normed.mean(axis=0), 0.0, atol=1e-9)
    assert normed.std() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no spread"):
        normalise_trajectory(np.ones((N, 2)))


# --- enroll / verify ------------------------------------------------------

def test_verify_genuine_passes_and_forgery_fails():
    verifier = _enrolled()
    genuine = verifier.verify("usman", make_signature("usman", seed=7, n_points=N))
    assert genuine["decision"] == "genuine"
    assert genuine["distance"] <= DEFAULT_THRESHOLD
    assert 0.0 < genuine["score"] <= 1.0
    forgery = verifier.verify("usman", make_signature("usman", genuine=False, seed=7, n_points=N))
    assert forgery["decision"] == "forgery"
    assert forgery["distance"] > DEFAULT_THRESHOLD
    assert forgery["score"] == 0.0
    assert genuine["distance"] < forgery["distance"]


def test_verify_result_fields():
    verifier = _enrolled()
    result = verifier.verify("usman", make_signature("usman", seed=7, n_points=N))
    assert result["person_id"] == "usman"
    assert result["threshold"] == DEFAULT_THRESHOLD
    assert result["n_references"] == 2
    assert set(result) == {"person_id", "distance", "threshold", "decision", "score", "n_references"}


def test_verify_before_enrolling_raises():
    verifier = SignatureVerifier()
    with pytest.raises(ValueError, match="no enrolled references"):
        verifier.verify("usman", make_signature("usman", seed=1, n_points=N))
    with pytest.raises(ValueError, match="no enrolled references"):
        verify(create_enrollment(), "usman", make_signature("usman", seed=1, n_points=N))


def test_verify_bad_threshold():
    verifier = _enrolled()
    sig = make_signature("usman", seed=7, n_points=N)
    for bad in (0, -0.5, float("nan"), float("inf"), "high"):
        with pytest.raises(ValueError, match="threshold"):
            verifier.verify("usman", sig, threshold=bad)
    with pytest.raises(ValueError, match="threshold"):
        SignatureVerifier(threshold=0)


def test_verify_custom_threshold_flips_decision():
    verifier = _enrolled()
    forgery = make_signature("usman", genuine=False, seed=7, n_points=N)
    assert verifier.verify("usman", forgery)["decision"] == "forgery"
    # A huge threshold accepts anything; a tiny one rejects everything.
    assert verifier.verify("usman", forgery, threshold=10.0)["decision"] == "genuine"
    genuine = make_signature("usman", seed=7, n_points=N)
    assert verifier.verify("usman", genuine, threshold=1e-6)["decision"] == "forgery"


def test_enroll_accepts_single_trajectory_and_rejects_empty():
    store = create_enrollment()
    enroll(store, "usman", make_signature("usman", seed=1, n_points=N))
    assert len(store["usman"]) == 1
    with pytest.raises(ValueError, match="must not be empty"):
        enroll(create_enrollment(), "usman", [])
    with pytest.raises(ValueError, match="too few points"):
        enroll(create_enrollment(), "usman", [np.zeros((5, 2))])
    with pytest.raises(ValueError, match="person_id"):
        enroll(create_enrollment(), "", [make_signature("usman", seed=1, n_points=N)])
    with pytest.raises(ValueError, match="store"):
        enroll([], "usman", [make_signature("usman", seed=1, n_points=N)])


def test_enroll_second_person_is_independent():
    verifier = _enrolled("usman")
    verifier.enroll("alice", [make_signature("alice", seed=1, n_points=N)])
    assert verifier.is_enrolled("alice")
    assert not verifier.is_enrolled("mallory")
    # Alice's own signature verifies for Alice, not for Usman.
    alice_sig = make_signature("alice", seed=9, n_points=N)
    assert verifier.verify("alice", alice_sig)["decision"] == "genuine"
    assert verifier.verify("usman", alice_sig)["decision"] == "forgery"


def test_verify_too_few_points_raises():
    verifier = _enrolled()
    with pytest.raises(ValueError, match="too few points"):
        verifier.verify("usman", np.zeros((5, 2)))
