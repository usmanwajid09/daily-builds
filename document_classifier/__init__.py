"""Public API for the document_classifier package."""

from .classifier import (
    CLASSES,
    FEATURE_NAMES,
    HEIGHT,
    WIDTH,
    DocumentClassifier,
    accuracy,
    confusion_matrix,
    extract_features,
    fit,
    make_document,
    predict,
    predict_proba,
    train_classifier,
)

__all__ = [
    "CLASSES",
    "FEATURE_NAMES",
    "HEIGHT",
    "WIDTH",
    "DocumentClassifier",
    "accuracy",
    "confusion_matrix",
    "extract_features",
    "fit",
    "make_document",
    "predict",
    "predict_proba",
    "train_classifier",
]
