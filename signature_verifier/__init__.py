"""Signature verification from synthetic pen-stroke trajectories."""

from signature_verifier.verifier import (
    DEFAULT_THRESHOLD,
    MIN_POINTS,
    SignatureVerifier,
    create_enrollment,
    dtw_distance,
    enroll,
    extract_features,
    make_signature,
    normalise_trajectory,
    verify,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "MIN_POINTS",
    "SignatureVerifier",
    "create_enrollment",
    "dtw_distance",
    "enroll",
    "extract_features",
    "make_signature",
    "normalise_trajectory",
    "verify",
]
