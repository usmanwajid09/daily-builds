import pytest

from markov_chain_text import MarkovChainText

CORPUS = (
    "machine learning models learn patterns from data "
    "machine learning engineers build models that predict outcomes "
    "data scientists collect data and train models on computers "
    "models improve when they see more data and better features "
    "machine learning is useful for prediction and classification"
)


def _fitted(**kwargs):
    chain = MarkovChainText(**kwargs)
    chain.fit(CORPUS)
    return chain


def test_generate_returns_requested_length():
    chain = _fitted(order=2, random_state=42)
    text = chain.generate(length=30, seed_words="machine learning")
    assert len(text.split()) == 30  # headline number from the demo


def test_generate_starts_with_seed_words():
    chain = _fitted(order=2, random_state=42)
    text = chain.generate(length=10, seed_words="machine learning")
    assert text.split()[:2] == ["machine", "learning"]


def test_deterministic_with_same_seed():
    chain_a = _fitted(order=2, random_state=42)
    chain_b = _fitted(order=2, random_state=42)
    assert chain_a.generate(length=20, seed_words="machine learning") == (
        chain_b.generate(length=20, seed_words="machine learning")
    )


def test_different_seeds_can_differ():
    chain_a = _fitted(order=2, random_state=1)
    chain_b = _fitted(order=2, random_state=2)
    assert chain_a.generate(length=30) != chain_b.generate(length=30)


def test_generated_words_all_from_corpus():
    chain = _fitted(order=2, random_state=7)
    vocab = set(CORPUS.split())
    text = chain.generate(length=50, seed_words="data scientists")
    assert all(word in vocab for word in text.split())


def test_generate_without_seed_starts_from_corpus_start():
    chain = _fitted(order=2, random_state=42)
    text = chain.generate(length=5)
    assert text.split()[:2] == ["machine", "learning"]


def test_per_call_seed_overrides_instance_seed():
    chain = _fitted(order=2, random_state=42)
    text_a = chain.generate(length=20, random_state=1)
    text_b = chain.generate(length=20, random_state=1)
    assert text_a == text_b


def test_order_one_works():
    chain = MarkovChainText(order=1, random_state=0)
    chain.fit("a b a b a c b a")
    text = chain.generate(length=8, seed_words="a")
    assert len(text.split()) == 8
    assert text.split()[0] == "a"


def test_n_states_positive():
    chain = _fitted(order=2, random_state=42)
    assert chain.n_states > 0


def test_fit_returns_self_and_reports_states():
    chain = MarkovChainText(order=2, random_state=42)
    assert chain.fit(CORPUS) is chain
    assert chain.n_states == len(set(
        tuple(CORPUS.split()[i:i + 2]) for i in range(len(CORPUS.split()) - 2)
    ))


def test_empty_text_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        MarkovChainText().fit("   ")


def test_non_string_text_raises():
    with pytest.raises(TypeError, match="must be a string"):
        MarkovChainText().fit(123)


def test_too_short_text_raises():
    with pytest.raises(ValueError, match="more than order"):
        MarkovChainText(order=2).fit("only two")


def test_generate_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        MarkovChainText().generate(length=5)


def test_wrong_seed_word_count_raises():
    chain = _fitted(order=2, random_state=42)
    with pytest.raises(ValueError, match="exactly 2 words"):
        chain.generate(length=10, seed_words="machine")


def test_unknown_seed_words_raise():
    chain = _fitted(order=2, random_state=42)
    with pytest.raises(ValueError, match="never occur"):
        chain.generate(length=10, seed_words="quantum banana")


def test_bad_order_raises():
    with pytest.raises(ValueError, match="order"):
        MarkovChainText(order=0)


def test_bad_length_raises():
    chain = _fitted(order=2, random_state=42)
    with pytest.raises(ValueError, match="length"):
        chain.generate(length=0)


def test_seed_words_as_list_accepted():
    chain = _fitted(order=2, random_state=42)
    text = chain.generate(length=6, seed_words=["machine", "learning"])
    assert text.split()[:2] == ["machine", "learning"]
