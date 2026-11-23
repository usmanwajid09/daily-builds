"""HOG features + kNN classifier for synthetic 8x8 digit images."""

from digit_classifier_hog.hog_digits import (
    DEFAULT_CELL_SIZE,
    DEFAULT_K,
    DEFAULT_N_BINS,
    IMAGE_SHAPE,
    HOGKNNClassifier,
    accuracy,
    hog_features,
    hog_vector_length,
    make_dataset,
    make_images,
    predict,
    template_match_accuracy,
    train,
)

__all__ = [
    "DEFAULT_CELL_SIZE",
    "DEFAULT_K",
    "DEFAULT_N_BINS",
    "IMAGE_SHAPE",
    "HOGKNNClassifier",
    "accuracy",
    "hog_features",
    "hog_vector_length",
    "make_dataset",
    "make_images",
    "predict",
    "template_match_accuracy",
    "train",
]
