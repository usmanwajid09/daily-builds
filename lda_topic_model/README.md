# LDA Topic Model

Latent Dirichlet Allocation, from scratch, fitted with **collapsed Gibbs
sampling**. Each document is a mixture of topics and each topic is a
distribution over words. Only numpy and the standard library are used.

## Usage

```python
from lda_topic_model.model import LatentDirichletAllocation

docs = [
    "the striker scored a brilliant goal in the final match",
    "chop the garlic and onion then fry them in olive oil",
    # ...
]

lda = LatentDirichletAllocation(
    n_topics=2, alpha=0.1, beta=0.1, n_iter=500, random_state=42
).fit(docs)

lda.top_words(0, n=8)        # most probable words of topic 0
lda.doc_topic_dist(0)        # topic mixture of document 0 (sums to 1)
lda.transform(["the striker scored the winning goal"])  # new documents
```

## How it works

Collapsed Gibbs sampling: each word token is assigned a topic, then
repeatedly re-sampled from

    P(topic k | rest) ∝ (n_doc,k + α) · (n_k,word + β) / (n_k + V·β)

where `n_doc,k` counts topic `k` in the document, `n_k,word` counts the
word under topic `k`, and `V` is the vocabulary size. After the sweeps,
the smoothed counts give the topic-word (`phi`) and document-topic
(`theta`) distributions. A fixed seed (`random_state=42`) makes fits
deterministic.

`transform()` folds new documents in with the same sampler while holding
the fitted topic-word distributions fixed; documents with no known words
fall back to a uniform distribution.

## Validation

- `n_topics` must be an integer ≥ 2; `alpha` and `beta` must be > 0;
  `n_iter` must be an integer ≥ 1.
- `fit()` rejects an empty corpus, non-string documents, and documents
  with no word tokens.
- `top_words()`, `doc_topic_dist()`, and `transform()` raise `ValueError`
  before fitting and on out-of-range indices.

## Demo output

`python lda_topic_model/demo.py` fits 2 topics on 16 documents — 8 about
sports, 8 about cooking:

```
Corpus: 16 documents, vocabulary: 96 words
Topics: 2, Gibbs sweeps: 500
Topic 0: the, a, match, goal, striker, in, penalty, championship
Topic 1: the, and, in, oven, garlic, sauce, bake, add

sports doc 0    -> topic 0: 0.99, topic 1: 0.01
sports doc 4    -> topic 0: 0.99, topic 1: 0.01
cooking doc 0   -> topic 0: 0.01, topic 1: 0.99
cooking doc 12  -> topic 0: 0.01, topic 1: 0.99

'the striker scored the winning goal in the stadium' -> topic 0: 0.99, topic 1: 0.01
'bake the cake in the oven with butter and sugar' -> topic 0: 0.01, topic 1: 0.99

All topic-word rows sum to 1: True
All doc-topic rows sum to 1: True
```

The two themes separate cleanly: topic 0 holds the sports words
(`match`, `goal`, `striker`, `penalty`, `championship`) and topic 1 the
cooking words (`oven`, `garlic`, `sauce`, `bake`). Every document sits
~99% in its own theme's topic, and unseen documents are assigned to the
right topic too.

## Tests

30 tests in `tests/test_lda_topic_model.py`: constructor and `fit()`
validation, unfitted/bad-index errors, distribution shapes and
normalisation, determinism, and the sports-vs-cooking separation
(including `transform()` on new documents).
