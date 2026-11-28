"""Tests for logreg_text_classifier (TF-IDF + logistic regression)."""

import numpy as np
import pytest

from logreg_text_classifier import (
    LogisticRegressionTextClassifier,
    TfidfVectorizer,
    tokenize,
)
from logreg_text_classifier.demo import DATASET, split_by_class


def make_clf(**kwargs):
    params = {"learning_rate": 0.5, "epochs": 300, "reg": 0.01,
              "random_state": 0}
    params.update(kwargs)
    return LogisticRegressionTextClassifier(**params)


@pytest.fixture()
def fitted_clf():
    train, _ = split_by_class(DATASET)
    clf = make_clf()
    clf.fit([doc for doc, _ in train], [label for _, label in train])
    return clf


def test_fit_predict_round_trip(fitted_clf):
    preds = fitted_clf.predict([
        "the striker scored a goal in the football match",
        "bake the cake in the oven with sugar and eggs",
        "the python code runs the algorithm on the server",
    ])
    assert preds == ["sports", "cooking", "technology"]


def test_fit_returns_self():
    clf = make_clf()
    assert clf.fit(["goal football match", "bake oven cake"],
                   ["sports", "cooking"]) is clf


def test_accuracy_above_chance_on_demo_data():
    train, test = split_by_class(DATASET)
    clf = make_clf()
    clf.fit([doc for doc, _ in train], [label for _, label in train])
    acc = clf.score([doc for doc, _ in test], [label for _, label in test])
    assert acc >= 0.80  # chance level is 1/3 for 3 classes


def test_predict_proba_rows_sum_to_one(fitted_clf):
    probs = fitted_clf.predict_proba([
        "the striker scored a goal",
        "bake the bread in the oven",
        "the software runs on the server",
    ])
    assert probs.shape == (3, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), atol=1e-8)


def test_probabilities_in_zero_one(fitted_clf):
    probs = fitted_clf.predict_proba(["goal football match",
                                      "garlic tomato sauce",
                                      "database server code"])
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_unknown_word_handling(fitted_clf):
    # Only out-of-vocabulary words: must still return a valid label and
    # a valid probability distribution (bias carries the prediction).
    preds = fitted_clf.predict(["zqxj wobble flibberty gibbet"])
    assert preds[0] in {"sports", "cooking", "technology"}
    probs = fitted_clf.predict_proba(["zqxj wobble flibberty gibbet"])
    np.testing.assert_allclose(probs.sum(axis=1), [1.0], atol=1e-8)


def test_empty_training_data_raises():
    with pytest.raises(ValueError):
        make_clf().fit([], [])


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        make_clf().fit(["goal football", "bake oven cake"], ["sports"])


def test_predict_before_fit_raises():
    clf = make_clf()
    with pytest.raises(RuntimeError):
        clf.predict(["goal football match"])
    with pytest.raises(RuntimeError):
        clf.predict_proba(["goal football match"])


def test_non_string_input_raises():
    with pytest.raises(TypeError):
        make_clf().fit(["goal football", 42], ["sports", "cooking"])
    clf = make_clf()
    clf.fit(["goal football match", "bake oven cake"], ["sports", "cooking"])
    with pytest.raises(TypeError):
        clf.predict([123])


def test_empty_document_raises():
    with pytest.raises(ValueError):
        make_clf().fit(["goal football", "   !!!  "],
                       ["sports", "cooking"])
    with pytest.raises(ValueError):
        make_clf().fit(["goal football", ""], ["sports", "cooking"])


def test_determinism_same_seed_same_predictions():
    train, test = split_by_class(DATASET)
    X_train = [doc for doc, _ in train]
    y_train = [label for _, label in train]
    X_test = [doc for doc, _ in test]
    clf_a = make_clf(random_state=42).fit(X_train, y_train)
    clf_b = make_clf(random_state=42).fit(X_train, y_train)
    assert clf_a.predict(X_test) == clf_b.predict(X_test)
    np.testing.assert_allclose(clf_a.predict_proba(X_test),
                               clf_b.predict_proba(X_test))


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, WORLD! Goal-123.") == ["hello", "world", "goal", "123"]
    with pytest.raises(TypeError):
        tokenize(42)


def test_vocabulary_from_training_data_only(fitted_clf):
    assert "goal" in fitted_clf.vectorizer_.vocabulary_
    assert "zqxj" not in fitted_clf.vectorizer_.vocabulary_
    # Rows are L2-normalised: a known document has unit norm.
    vec = fitted_clf.vectorizer_.transform(["goal football match"])
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-8
