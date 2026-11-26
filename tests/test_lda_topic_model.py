"""Tests for the lda_topic_model program."""

import numpy as np
import pytest

from lda_topic_model.model import LatentDirichletAllocation, tokenize
from lda_topic_model.demo import CORPUS, SPORTS_DOCS, COOKING_DOCS

SPORTS_WORDS = {"goal", "striker", "match", "team", "championship",
                "coach", "penalty", "referee", "stadium", "trophy"}
COOKING_WORDS = {"garlic", "oven", "sauce", "bake", "herbs", "recipe",
                 "onion", "flour", "butter", "simmer"}


def fit_demo(**kwargs):
    params = dict(n_topics=2, alpha=0.1, beta=0.1, n_iter=500,
                  random_state=42)
    params.update(kwargs)
    return LatentDirichletAllocation(**params).fit(CORPUS)


# --- constructor validation ---

def test_n_topics_below_two_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        LatentDirichletAllocation(n_topics=1)


def test_n_topics_zero_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        LatentDirichletAllocation(n_topics=0)


def test_n_topics_non_integer_rejected():
    with pytest.raises(TypeError, match="integer"):
        LatentDirichletAllocation(n_topics=2.5)


def test_alpha_not_positive_rejected():
    with pytest.raises(ValueError, match="alpha must be positive"):
        LatentDirichletAllocation(alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be positive"):
        LatentDirichletAllocation(alpha=-0.5)


def test_beta_not_positive_rejected():
    with pytest.raises(ValueError, match="beta must be positive"):
        LatentDirichletAllocation(beta=-1.0)


def test_n_iter_below_one_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        LatentDirichletAllocation(n_iter=0)


def test_n_iter_non_integer_rejected():
    with pytest.raises(TypeError, match="integer"):
        LatentDirichletAllocation(n_iter=10.5)


def test_random_state_non_integer_rejected():
    with pytest.raises(TypeError, match="integer or None"):
        LatentDirichletAllocation(random_state="seed")


# --- fit validation ---

def test_fit_empty_corpus_rejected():
    with pytest.raises(ValueError, match="at least one document"):
        LatentDirichletAllocation().fit([])


def test_fit_non_string_document_rejected():
    with pytest.raises(TypeError, match="must be a string"):
        LatentDirichletAllocation().fit(["a real doc", 123])


def test_fit_document_without_tokens_rejected():
    with pytest.raises(ValueError, match="at least one word token"):
        LatentDirichletAllocation().fit(["hello world", "!!! 123"])


def test_fit_returns_self():
    lda = LatentDirichletAllocation(n_iter=10)
    assert lda.fit(["the cat sat", "the dog ran"]) is lda


# --- unfitted / bad-index errors ---

def test_top_words_before_fit_rejected():
    with pytest.raises(ValueError, match="not fitted"):
        LatentDirichletAllocation().top_words(0)


def test_doc_topic_dist_before_fit_rejected():
    with pytest.raises(ValueError, match="not fitted"):
        LatentDirichletAllocation().doc_topic_dist(0)


def test_transform_before_fit_rejected():
    with pytest.raises(ValueError, match="not fitted"):
        LatentDirichletAllocation().transform(["the cat sat"])


def test_top_words_bad_topic_rejected():
    lda = fit_demo()
    with pytest.raises(ValueError, match="between 0 and 1"):
        lda.top_words(2)
    with pytest.raises(ValueError, match="between 0 and 1"):
        lda.top_words(-1)


def test_top_words_bad_n_rejected():
    lda = fit_demo()
    with pytest.raises(ValueError, match="at least 1"):
        lda.top_words(0, n=0)


def test_doc_topic_dist_bad_index_rejected():
    lda = fit_demo()
    with pytest.raises(ValueError, match="between 0 and 15"):
        lda.doc_topic_dist(16)
    with pytest.raises(ValueError, match="between 0 and 15"):
        lda.doc_topic_dist(-1)


def test_transform_empty_list_rejected():
    lda = fit_demo()
    with pytest.raises(ValueError, match="at least one document"):
        lda.transform([])


def test_transform_non_string_rejected():
    lda = fit_demo()
    with pytest.raises(TypeError, match="must be a string"):
        lda.transform([42])


# --- fitted shapes and normalisation ---

def test_fitted_shapes():
    lda = fit_demo()
    assert lda.topic_word_dist_.shape == (2, len(lda.vocab_))
    assert lda.doc_topic_dist_.shape == (len(CORPUS), 2)
    assert len(lda.vocab_) > 0


def test_distributions_sum_to_one():
    lda = fit_demo()
    assert np.allclose(lda.topic_word_dist_.sum(axis=1), 1.0)
    assert np.allclose(lda.doc_topic_dist_.sum(axis=1), 1.0)
    for i in range(len(CORPUS)):
        assert np.isclose(lda.doc_topic_dist(i).sum(), 1.0)


def test_top_words_returns_vocab_words():
    lda = fit_demo()
    words = lda.top_words(0, n=5)
    assert len(words) == 5
    assert all(w in lda.vocab_ for w in words)
    assert len(set(words)) == 5


def test_deterministic_fit():
    a = fit_demo()
    b = fit_demo()
    assert np.allclose(a.topic_word_dist_, b.topic_word_dist_)
    assert np.allclose(a.doc_topic_dist_, b.doc_topic_dist_)
    assert a.top_words(0, n=10) == b.top_words(0, n=10)


def test_different_seed_can_change_result():
    # Only checks the seed is actually used, not a specific outcome.
    a = fit_demo(random_state=42)
    b = fit_demo(random_state=7)
    assert a.random_state != b.random_state


# --- demo behaviour: sports vs cooking separation ---

def test_topics_separate_themes():
    lda = fit_demo()
    topic_words = [set(lda.top_words(k, n=10)) for k in range(2)]
    sports_hits = [len(t & SPORTS_WORDS) for t in topic_words]
    cooking_hits = [len(t & COOKING_WORDS) for t in topic_words]
    # One topic must be sports-heavy and the other cooking-heavy.
    assert max(sports_hits) >= 3
    assert max(cooking_hits) >= 3
    assert sports_hits.index(max(sports_hits)) != cooking_hits.index(
        max(cooking_hits))


def test_sports_docs_dominate_one_topic():
    lda = fit_demo()
    sports_dists = np.array([lda.doc_topic_dist(i)
                             for i in range(len(SPORTS_DOCS))])
    cooking_dists = np.array([lda.doc_topic_dist(i)
                              for i in range(len(SPORTS_DOCS),
                                             len(CORPUS))])
    sports_topic = int(np.argmax(sports_dists.mean(axis=0)))
    assert sports_dists[:, sports_topic].mean() > 0.8
    assert cooking_dists[:, sports_topic].mean() < 0.2


def test_transform_assigns_new_docs_to_right_topic():
    lda = fit_demo()
    dists = lda.transform([
        "the striker scored the winning goal in the stadium",
        "bake the cake in the oven with butter and sugar",
    ])
    assert dists.shape == (2, 2)
    assert np.allclose(dists.sum(axis=1), 1.0)
    sports_topic = int(np.argmax(
        np.mean([lda.doc_topic_dist(i) for i in range(len(SPORTS_DOCS))],
                axis=0)))
    assert dists[0, sports_topic] > 0.8
    assert dists[1, sports_topic] < 0.2


def test_transform_unknown_words_falls_back_to_uniform():
    lda = fit_demo()
    dist = lda.transform(["zxqv wjkl qmbp"])[0]
    assert np.allclose(dist, [0.5, 0.5])


def test_tokenize():
    assert tokenize("Hello, WORLD! 123") == ["hello", "world"]
    assert tokenize("") == []
    with pytest.raises(TypeError, match="must be a string"):
        tokenize(42)
