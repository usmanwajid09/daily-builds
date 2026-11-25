import pytest

from tfidf_vectorizer import TfidfVectorizer

DOCS = [
    "Cats sleep all day and chase mice in the garden",
    "Dogs love walks and bark at strangers in the park",
    "Machine learning models learn patterns from data",
    "Deep learning uses neural networks to classify images",
    "Cats and dogs are popular pets in many homes",
    "Training a model requires data and careful evaluation",
]


def _fitted():
    vec = TfidfVectorizer()
    vec.fit(DOCS)
    return vec


def test_search_ranks_ml_doc_first():
    vec = _fitted()
    results = vec.search("machine learning models", top_k=3)
    assert results[0][0] == 2  # headline number from the demo
    assert results[0][1] > 0.5


def test_search_returns_at_most_top_k():
    vec = _fitted()
    assert len(vec.search("cats dogs", top_k=2)) == 2
    assert len(vec.search("cats dogs", top_k=10)) == len(DOCS)


def test_scores_sorted_descending_and_in_range():
    vec = _fitted()
    results = vec.search("cats dogs data")
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_cat_query_prefers_cat_docs():
    vec = _fitted()
    results = vec.search("cats chase mice")
    assert results[0][0] == 0


def test_identical_document_scores_one():
    vec = _fitted()
    vectors = vec.transform([DOCS[2]])
    assert TfidfVectorizer.cosine_similarity(vectors[0], vectors[0]) == pytest.approx(1.0)


def test_unrelated_query_scores_zero():
    vec = _fitted()
    results = vec.search("quantum entanglement")
    assert all(score == 0.0 for _, score in results)


def test_fit_transform_matches_transform():
    vec = _fitted()
    fitted = vec.fit_transform(DOCS)
    again = vec.transform(DOCS)
    assert fitted == again


def test_vectors_are_l2_normalised():
    vec = _fitted()
    for vector in vec.transform(DOCS):
        norm_sq = sum(v * v for v in vector.values())
        assert norm_sq == pytest.approx(1.0)


def test_single_document_fits():
    vec = TfidfVectorizer()
    vec.fit(["only one document here"])
    assert len(vec.vocabulary_) == 4
    assert vec.search("only one", top_k=1)[0][0] == 0


def test_vocabulary_is_sorted_unique_terms():
    vec = _fitted()
    assert vec.vocabulary_ == sorted(set(vec.vocabulary_))
    assert "machine" in vec.vocabulary_
    assert "learning" in vec.vocabulary_


def test_empty_documents_raise():
    with pytest.raises(ValueError, match="must not be empty"):
        TfidfVectorizer().fit([])


def test_blank_document_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        TfidfVectorizer().fit(["a real doc", "   "])


def test_non_string_document_raises():
    with pytest.raises(TypeError, match="must be a string"):
        TfidfVectorizer().fit(["a real doc", 42])


def test_string_instead_of_list_raises():
    with pytest.raises(TypeError, match="list of strings"):
        TfidfVectorizer().fit("just one string")


def test_search_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        TfidfVectorizer().search("hello")


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        TfidfVectorizer().transform(["hello"])


def test_empty_query_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        _fitted().search("   ")


def test_bad_top_k_raises():
    with pytest.raises(ValueError, match="top_k"):
        _fitted().search("cats", top_k=0)


def test_cosine_similarity_with_empty_vector():
    assert TfidfVectorizer.cosine_similarity({}, {"a": 1.0}) == 0.0
    assert TfidfVectorizer.cosine_similarity({"a": 1.0}, {}) == 0.0
