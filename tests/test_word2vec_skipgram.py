import subprocess
import sys

import numpy as np
import pytest

from word2vec_skipgram.demo import CORPUS
from word2vec_skipgram.model import SkipGramWord2Vec, tokenize

SMALL = [
    "the dog eats food",
    "the cat eats food",
    "the dog drinks water",
    "the cat drinks water",
    "the dog sleeps at home",
    "the cat sleeps at home",
    "apple is sweet",
    "banana is sweet",
    "eat apple for breakfast",
    "eat banana for breakfast",
]


def _small_model(**kwargs):
    params = dict(
        embedding_dim=8, window=1, negative_samples=3,
        learning_rate=0.05, epochs=30, seed=42,
    )
    params.update(kwargs)
    return SkipGramWord2Vec(**params).fit(SMALL)


@pytest.fixture(scope="module")
def demo_model():
    return SkipGramWord2Vec(
        embedding_dim=16, window=1, negative_samples=5,
        learning_rate=0.05, epochs=200, seed=42,
    ).fit(CORPUS)


def test_fit_sets_vocab_and_embedding_shape():
    model = _small_model()
    assert set(model.vocab_) >= {"dog", "cat", "apple", "banana"}
    assert "the" not in model.vocab_  # stopwords are dropped
    assert model.embeddings_.shape == (len(model.vocab_), 8)
    assert model.n_pairs_ > 0
    assert len(model.loss_history_) == 30


def test_loss_decreases_over_training():
    model = _small_model()
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_same_seed_is_deterministic():
    a = _small_model()
    b = _small_model()
    assert np.allclose(a.embeddings_, b.embeddings_)


def test_different_seed_gives_different_embeddings():
    a = _small_model(seed=42)
    b = _small_model(seed=7)
    assert not np.allclose(a.embeddings_, b.embeddings_)


def test_dog_nearest_neighbour_is_cat(demo_model):
    neighbours = demo_model.most_similar("dog", topn=5)
    assert neighbours[0][0] == "cat"
    assert neighbours[0][1] > 0.9


def test_cat_nearest_neighbour_is_dog(demo_model):
    assert demo_model.most_similar("cat", topn=1)[0][0] == "dog"


def test_apple_nearest_neighbour_is_banana(demo_model):
    neighbours = demo_model.most_similar("apple", topn=5)
    assert neighbours[0][0] == "banana"
    assert neighbours[0][1] > 0.9


def test_similarity_dog_cat_beats_dog_apple(demo_model):
    assert demo_model.similarity("dog", "cat") > 0.9
    assert demo_model.similarity("dog", "cat") > demo_model.similarity("dog", "apple")
    assert demo_model.similarity("dog", "apple") < 0.5


def test_similarity_is_symmetric_and_bounded(demo_model):
    sim = demo_model.similarity("dog", "cat")
    assert demo_model.similarity("cat", "dog") == pytest.approx(sim)
    assert -1.0 <= sim <= 1.0
    assert demo_model.similarity("dog", "dog") == pytest.approx(1.0)


def test_analogy_puppy_dog_cat_is_kitten(demo_model):
    result = demo_model.analogy("puppy", "dog", "cat", topn=3)
    assert result[0][0] == "kitten"
    assert result[0][1] > 0.9


def test_most_similar_excludes_query_word_and_respects_topn(demo_model):
    neighbours = demo_model.most_similar("dog", topn=3)
    assert len(neighbours) == 3
    assert all(w != "dog" for w, _ in neighbours)
    scores = [s for _, s in neighbours]
    assert scores == sorted(scores, reverse=True)


def test_get_vector_shape_and_copy():
    model = _small_model()
    vec = model.get_vector("dog")
    assert vec.shape == (8,)
    vec[0] = 999.0
    assert model.get_vector("dog")[0] != 999.0  # returned copy, not a view


def test_contains_operator():
    model = _small_model()
    assert "dog" in model
    assert "zebra" not in model


def test_fit_accepts_token_lists():
    model = SkipGramWord2Vec(epochs=5, seed=42).fit(
        [["dog", "eats", "food"], ["cat", "eats", "food"]]
    )
    assert "dog" in model.vocab_ and "cat" in model.vocab_


def test_min_count_drops_rare_words():
    model = SkipGramWord2Vec(epochs=5, min_count=2, seed=42).fit(
        ["the dog eats food", "the cat eats food", "the dog is happy"]
    )
    assert "happy" not in model.vocab_  # appears only once
    assert "dog" in model.vocab_


def test_tokenize_lowercases_stems_and_strips_punctuation():
    # tokenize keeps stopwords (only fit() drops them); it lowercases,
    # strips punctuation and stems plurals.
    assert tokenize("Dogs Eat, Food!") == ["dog", "eat", "food"]
    assert tokenize("The Dogs Eat Food") == ["the", "dog", "eat", "food"]


def test_tokenize_non_string_raises():
    with pytest.raises(ValueError, match="must be a string"):
        tokenize(123)


def test_invalid_hyperparameters_raise():
    with pytest.raises(ValueError):
        SkipGramWord2Vec(embedding_dim=0)
    with pytest.raises(ValueError):
        SkipGramWord2Vec(window=0)
    with pytest.raises(ValueError):
        SkipGramWord2Vec(negative_samples=0)
    with pytest.raises(ValueError):
        SkipGramWord2Vec(learning_rate=0)
    with pytest.raises(ValueError):
        SkipGramWord2Vec(epochs=0)
    with pytest.raises(ValueError):
        SkipGramWord2Vec(min_count=0)
    with pytest.raises(TypeError):
        SkipGramWord2Vec(embedding_dim="16")


def test_fit_empty_sentences_raises():
    with pytest.raises(ValueError, match="at least one sentence"):
        SkipGramWord2Vec().fit([])


def test_fit_single_string_raises():
    with pytest.raises(TypeError, match="list of strings"):
        SkipGramWord2Vec().fit("the dog eats food")


def test_fit_no_usable_tokens_raises():
    with pytest.raises(ValueError, match="no usable tokens"):
        SkipGramWord2Vec().fit(["the a an", "123 !!!"])


def test_fit_single_word_vocab_raises():
    with pytest.raises(ValueError, match="at least 2 distinct"):
        SkipGramWord2Vec().fit(["dog dog dog", "dog dog"])


def test_fit_too_short_sentences_raise():
    with pytest.raises(ValueError, match="no co-occurrence pairs"):
        SkipGramWord2Vec().fit(["dog", "cat", "bird"])


def test_fit_non_string_sentence_raises():
    with pytest.raises(TypeError, match="must be a string or a list"):
        SkipGramWord2Vec().fit(["the dog eats", 42])


def test_methods_before_fit_raise():
    model = SkipGramWord2Vec()
    with pytest.raises(ValueError, match="not fitted"):
        model.most_similar("dog")
    with pytest.raises(ValueError, match="not fitted"):
        model.similarity("dog", "cat")
    with pytest.raises(ValueError, match="not fitted"):
        model.get_vector("dog")
    with pytest.raises(ValueError, match="not fitted"):
        model.analogy("dog", "cat", "bird")


def test_unknown_word_raises():
    model = _small_model()
    with pytest.raises(ValueError, match="not in the vocabulary"):
        model.most_similar("zebra")
    with pytest.raises(ValueError, match="not in the vocabulary"):
        model.similarity("dog", "zebra")
    with pytest.raises(ValueError, match="not in the vocabulary"):
        model.get_vector("zebra")
    with pytest.raises(ValueError, match="not in the vocabulary"):
        model.analogy("dog", "cat", "zebra")


def test_invalid_topn_raises():
    model = _small_model()
    with pytest.raises(ValueError):
        model.most_similar("dog", topn=0)
    with pytest.raises(ValueError):
        model.analogy("dog", "cat", "apple", topn=-2)
    with pytest.raises(TypeError):
        model.most_similar("dog", topn=2.5)


def test_demo_runs_standalone_and_shows_dog_near_cat():
    result = subprocess.run(
        [sys.executable, "word2vec_skipgram/demo.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "Nearest to 'dog': cat (0.94)" in result.stdout
    assert "kitten (0.95)" in result.stdout
