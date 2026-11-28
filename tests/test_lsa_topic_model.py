"""Tests for LSA topic modelling (numpy-only TF-IDF + truncated SVD)."""

import numpy as np
import pytest

from lsa_topic_model import LSATopicModel
from lsa_topic_model.demo import COOKING, PETS, SPACE

DOCUMENTS = PETS + COOKING + SPACE
LABELS = ["pets"] * 4 + ["cooking"] * 4 + ["space"] * 4


def test_fit_returns_self_and_topics_have_right_shape():
    model = LSATopicModel(n_topics=3)
    assert model.fit(DOCUMENTS) is model
    topics = model.get_topics(n_words=5)
    assert len(topics) == 3
    assert all(len(topic) == 5 for topic in topics)


def test_top_words_are_strings_with_finite_weights():
    model = LSATopicModel(n_topics=3).fit(DOCUMENTS)
    for topic in model.get_topics(n_words=5):
        for term, weight in topic:
            assert isinstance(term, str) and term
            assert isinstance(weight, float)
            assert np.isfinite(weight)
        weights = [abs(w) for _, w in topic]
        assert weights == sorted(weights, reverse=True)


def test_transform_shape():
    model = LSATopicModel(n_topics=3).fit(DOCUMENTS)
    projected = model.transform(DOCUMENTS)
    assert projected.shape == (12, 3)


def test_transform_rows_are_finite():
    model = LSATopicModel(n_topics=3).fit(DOCUMENTS)
    projected = model.transform(DOCUMENTS)
    assert np.all(np.isfinite(projected))
    # Unseen documents also project without error.
    new_docs = ["a playful kitten chased the dog", "the chef baked bread"]
    assert model.transform(new_docs).shape == (2, 3)
    assert np.all(np.isfinite(model.transform(new_docs)))


def test_explained_variance_ratio_positive_and_sums_to_at_most_one():
    model = LSATopicModel(n_topics=3).fit(DOCUMENTS)
    ratios = model.explained_variance_ratio_
    assert ratios.shape == (3,)
    assert np.all(ratios > 0)
    assert ratios.sum() <= 1.0 + 1e-12
    assert np.all(np.diff(ratios) <= 1e-12)  # non-increasing


def test_demo_themes_group_into_distinct_dominant_topics():
    model = LSATopicModel(n_topics=3).fit(DOCUMENTS)
    dominant = model.dominant_topics(DOCUMENTS)
    theme_topic = {}
    for theme in ("pets", "cooking", "space"):
        topics = [int(dominant[i]) for i, lab in enumerate(LABELS) if lab == theme]
        assert len(set(topics)) == 1, f"{theme} docs split across topics {topics}"
        theme_topic[theme] = topics[0]
    assert len(set(theme_topic.values())) == 3, "themes must map to distinct topics"


def test_reconstruction_error_decreases_with_more_topics():
    errors = [
        LSATopicModel(n_topics=k).fit(DOCUMENTS).reconstruction_error()
        for k in (1, 2, 3)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert all(0.0 <= e <= 1.0 for e in errors)


def test_empty_documents_raise_value_error():
    with pytest.raises(ValueError):
        LSATopicModel(n_topics=2).fit([])


def test_documents_with_no_tokens_raise_value_error():
    with pytest.raises(ValueError):
        LSATopicModel(n_topics=2).fit(["!!!", "   "])


def test_n_topics_too_large_raises_value_error():
    # min(n_docs, n_terms) is small for a 2-doc corpus.
    with pytest.raises(ValueError):
        LSATopicModel(n_topics=2).fit(["cats and dogs", "stars and planets"])


def test_n_topics_below_one_raises_value_error():
    with pytest.raises(ValueError):
        LSATopicModel(n_topics=0)


def test_non_string_documents_raise_type_error():
    with pytest.raises(TypeError):
        LSATopicModel(n_topics=2).fit(["cats and dogs", 42])
    with pytest.raises(TypeError):
        LSATopicModel(n_topics=2).fit("a single string is not a corpus")


def test_transform_before_fit_raises_runtime_error():
    with pytest.raises(RuntimeError):
        LSATopicModel(n_topics=2).transform(["cats and dogs"])


def test_get_topics_before_fit_raises_runtime_error():
    with pytest.raises(RuntimeError):
        LSATopicModel(n_topics=2).get_topics()


def test_topic_coherence_returns_finite_scores():
    model = LSATopicModel(n_topics=3).fit(DOCUMENTS)
    scores = model.topic_coherence(n_words=5)
    assert len(scores) == 3
    assert all(np.isfinite(s) for s in scores)
