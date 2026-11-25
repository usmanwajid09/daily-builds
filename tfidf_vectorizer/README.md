# TF-IDF Vectorizer

A TF-IDF vectorizer plus cosine-similarity document search, implemented
with the **standard library only** — no numpy, no sklearn.

## How it works

Documents are tokenised to lowercase alphanumeric words. Each document
becomes a sparse `{term: weight}` vector where

- **TF** (term frequency) is `1 + log(count)` for terms that appear,
- **IDF** (inverse document frequency) is `log((1 + N) / (1 + df)) + 1`,
  so rare terms count more and common terms count less,
- vectors are **L2-normalised**, so cosine similarity is a plain dot
  product and always lies in [0, 1].

`search(query, top_k=3)` vectorises the query the same way and returns
the fitted documents ranked by cosine similarity as `(doc_index, score)`
pairs, best first.

## Usage

```python
from tfidf_vectorizer import TfidfVectorizer

docs = [
    "Cats sleep all day and chase mice",
    "Machine learning models learn patterns from data",
    "Training a model requires data and careful evaluation",
]
vec = TfidfVectorizer()
vec.fit(docs)
print(vec.search("machine learning models", top_k=2))
# [(1, 0.72...), (2, 0.15...)] — doc 1 is the best match

vectors = vec.transform(["a brand new document"])  # sparse dicts
```

Fitting an empty list, blank documents, or non-string documents, and
searching with an empty query or before fitting, raise
`ValueError`/`TypeError`/`RuntimeError` with a clear message.

## Run the demo

```bash
python tfidf_vectorizer/demo.py
```

Searches six short documents for "machine learning models" and prints
the top 3 hits — the machine-learning document ranks first.

## Run the tests

```bash
pytest tests/test_tfidf_vectorizer.py -q
```
