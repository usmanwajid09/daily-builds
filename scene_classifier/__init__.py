"""Scene classifier: colour-histogram features + kNN, numpy only."""

from .scene_classifier import (
    SCENE_KINDS,
    SceneClassifier,
    extract_features,
    generate_scene,
    predict,
    train_classifier,
)

__all__ = [
    "SCENE_KINDS",
    "SceneClassifier",
    "extract_features",
    "generate_scene",
    "predict",
    "train_classifier",
]
