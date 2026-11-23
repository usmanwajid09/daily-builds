"""Synthetic leaf-disease detection with colour features and kNN."""

from leaf_disease_detector.leaves import (
    CLASSES,
    N_FEATURES,
    LeafDiseaseClassifier,
    accuracy,
    extract_features,
    make_dataset,
    make_leaf,
)

__all__ = [
    "CLASSES",
    "N_FEATURES",
    "LeafDiseaseClassifier",
    "accuracy",
    "extract_features",
    "make_dataset",
    "make_leaf",
]
