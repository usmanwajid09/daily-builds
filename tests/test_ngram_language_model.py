"""Tests for the n-gram language model."""

import math

import pytest

from ngram_language_model import NGramLanguageModel

CORPUS = [
    "the cat sits on the mat",
    "the dog runs in the park",
    "the cat chases the dog",
    "the dog chases the cat home",
    "my cat likes milk and fish",
    "my dog likes bones and walks",
    "the cat sleeps all day",
    "the dog barks at the cat outside",
    "a happy dog wags its tail",
    "a sleepy cat naps in the sun",
]


@pytest.fixture
def model():
    return NGramLanguageModel(order=2, k=0.1).fit(CORPUS)


def test_conditional_probs_sum_to_one(model):
    for context in [(), ("the",), ("cat",), ("dog",), ("zzzz",)]:
        total = sum(model.cond_prob(w, context) for w in model.vocab)
        assert total == pytest.approx(1.0, abs=1e-9)


def test_perplexity_in_domain_lower_than_out_of_domain(model):
    in_domain = model.perplexity(["the dog chases the cat"])
    out_domain = model.perplexity(["quantum banana spaceship zebra"])
    assert in_domain < out_domain
    assert in_domain < 10
    assert out_domain > 20


def test_perplexity_of_training_text_finite_and_positive(model):
    ppl = model.perplexity(CORPUS)
    assert math.isfinite(ppl) and ppl > 1.0


def test_prob_and_log_prob_agree(model):
    sentence = "the cat sleeps all day"
    assert model.prob(sentence) == pytest.approx(math.exp(model.log_prob(sentence)))
    assert 0.0 < model.prob(sentence) <= 1.0


def test_generate_returns_text_starting_with_seed(model):
    text = model.generate("the cat", max_len=10)
    assert text.startswith("the cat")
    assert len(text.split()) > len("the cat".split())


def test_generate_deterministic_with_same_seed(model):
    first = model.generate("my dog", max_len=10)
    second = model.generate("my dog", max_len=10)
    assert first == second


def test_generate_handles_unknown_seed_words(model):
    text = model.generate("quantum banana", max_len=10)
    assert isinstance(text, str) and len(text.split()) > 0


def test_add_k_smoothing_gives_nonzero_prob_to_unseen_words(model):
    assert model.cond_prob("zebra", ("the",)) > 0.0
    assert model.prob("quantum banana spaceship") > 0.0
    assert math.isfinite(model.log_prob("quantum banana spaceship"))


def test_bad_order_rejected():
    with pytest.raises(ValueError):
        NGramLanguageModel(order=0)
    with pytest.raises(ValueError):
        NGramLanguageModel(order=4)


def test_bad_k_rejected():
    with pytest.raises(ValueError):
        NGramLanguageModel(k=0)
    with pytest.raises(ValueError):
        NGramLanguageModel(k=-0.5)


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        NGramLanguageModel().fit([])
    with pytest.raises(ValueError):
        NGramLanguageModel().fit(["   "])


def test_predict_and_generate_before_fit_raise():
    fresh = NGramLanguageModel()
    with pytest.raises(RuntimeError):
        fresh.log_prob("the cat")
    with pytest.raises(RuntimeError):
        fresh.perplexity(["the cat"])
    with pytest.raises(RuntimeError):
        fresh.generate("the cat")


def test_empty_sentence_rejected(model):
    with pytest.raises(ValueError):
        model.log_prob("")
    with pytest.raises(ValueError):
        model.perplexity([])


def test_order_one_model_works():
    uni = NGramLanguageModel(order=1, k=0.1).fit(CORPUS)
    assert uni.vocab_size > 0
    assert math.isfinite(uni.perplexity(["the cat"]))
    text = uni.generate("cat", max_len=5)
    assert text.startswith("cat") and len(text.split()) >= 1


def test_order_three_model_works():
    tri = NGramLanguageModel(order=3, k=0.1).fit(CORPUS)
    assert tri.perplexity(["the dog chases the cat"]) < tri.perplexity(
        ["quantum banana spaceship zebra"])
    assert tri.generate("my cat", max_len=10).startswith("my cat")


def test_sampled_generate_is_reproducible_with_seed(model):
    a = model.generate("the cat", max_len=10, sample=True, random_state=42)
    b = model.generate("the cat", max_len=10, sample=True, random_state=42)
    assert a == b
    assert a.startswith("the cat")
