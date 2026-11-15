import numpy as np
import pytest

from naive_bayes_text.model import MultinomialNB, tokenize

TRAIN_TEXTS = [
    "the team won the match with a great goal",
    "the striker scored two goals in the final",
    "the championship game ended in a draw",
    "the player received a red card",
    "the coach praised the team after the win",
    "the new phone has a faster processor",
    "the software update fixes the battery bug",
    "the laptop screen and keyboard are excellent",
    "the app crashes when opening the camera",
    "the developer released a new version today",
]
TRAIN_LABELS = ["sports"] * 5 + ["tech"] * 5


@pytest.fixture
def clf():
    return MultinomialNB(alpha=1.0).fit(TRAIN_TEXTS, TRAIN_LABELS)


def test_fit_sets_vocab_and_classes(clf):
    assert set(clf.classes_) == {"sports", "tech"}
    assert "goal" in clf.vocab_
    assert "processor" in clf.vocab_
    assert len(clf.vocab_) > 20


def test_training_accuracy_is_perfect(clf):
    assert clf.score(TRAIN_TEXTS, TRAIN_LABELS) == 1.0


def test_predict_unseen_sports_and_tech(clf):
    preds = clf.predict([
        "the goalkeeper saved the penalty",
        "the new tablet has a bright display",
    ])
    assert list(preds) == ["sports", "tech"]


def test_predict_proba_rows_sum_to_one(clf):
    probs = clf.predict_proba(["the team won", "the phone update"])
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_predict_log_proba_shape(clf):
    log_p = clf.predict_log_proba(["the match"])
    assert log_p.shape == (1, 2)
    assert np.isfinite(log_p).all()


def test_unknown_words_do_not_crash(clf):
    pred = clf.predict(["zebra quantum foxtrot"])
    assert pred[0] in ("sports", "tech")


def test_empty_document_predicts_prior_class(clf):
    pred = clf.predict([""])
    assert pred[0] in ("sports", "tech")


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, WORLD! 123 test-case") == ["hello", "world", "test", "case"]


def test_tokenize_non_string_raises():
    with pytest.raises(ValueError):
        tokenize(123)


def test_invalid_alpha_zero():
    with pytest.raises(ValueError):
        MultinomialNB(alpha=0)


def test_empty_training_raises():
    with pytest.raises(ValueError):
        MultinomialNB().fit([], [])


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        MultinomialNB().fit(["hello world"], ["a", "b"])


def test_no_tokens_raises():
    with pytest.raises(ValueError):
        MultinomialNB().fit(["123 !!!"], ["a"])


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        MultinomialNB().predict(["hello"])


def test_score_mismatched_labels_raises(clf):
    with pytest.raises(ValueError):
        clf.score(["hello world"], ["sports", "tech"])
