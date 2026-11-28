# LSA Topic Model

Latent Semantic Analysis (LSA) topic modelling, built from scratch with
**numpy only** — no scikit-learn, no new dependencies.

## How it works

1. **TF-IDF** — each document becomes a row of a (documents × terms)
   matrix. Tokens are lowercased alphanumeric words with common English
   stopwords removed. Term frequency is sublinear, `1 + log(count)`;
   inverse document frequency is smoothed, `log((1 + N) / (1 + df)) + 1`;
   each row is L2-normalised.
2. **Truncated SVD** — `numpy.linalg.svd` factorises the TF-IDF matrix
   and the top `n_topics` singular vectors are kept. Each topic is one
   row of `V`: a loading over the vocabulary, so its largest-magnitude
   words are the topic's most characteristic terms.
3. **Topics & projections** — `get_topics()` lists the top words per
   topic; `transform()` projects documents into topic space by dotting
   their TF-IDF vectors with the topic loadings. A document's dominant
   topic is the one with the largest absolute loading (SVD signs are
   arbitrary, magnitudes are what matter).

`explained_variance_ratio_` (squared singular values, normalised) says
how much of the TF-IDF structure each topic captures, and
`reconstruction_error()` falls as more topics are kept. `topic_coherence()`
scores each topic with a UMass-style co-occurrence measure on the
training corpus.

## Usage

```python
from lsa_topic_model import LSATopicModel

docs = [
    "My pet cat and dog play together in the garden every morning",
    "The chef baked bread in the oven and served it hot with butter",
    "The telescope showed a galaxy of stars beyond our planet",
    # ... more documents
]

model = LSATopicModel(n_topics=3).fit(docs)

for i, topic in enumerate(model.get_topics(n_words=5)):
    print(i, [term for term, _ in topic])

print(model.explained_variance_ratio_)
print(model.transform(docs).shape)   # (n_docs, 3)
print(model.dominant_topics(docs))   # dominant topic per document
```

## Example output

Run the demo from the repo root:

```
python lsa_topic_model/demo.py
```

Actual output (12 documents: 4 pets, 4 cooking, 4 space/astronomy):

```
LSA topics (top 5 words per topic):
  Topic 0: orbit (-0.394), stars (-0.392), moon (-0.306), rocket (-0.302), galaxy (-0.287)
  Topic 1: sauce (-0.410), garlic (-0.358), tomato (-0.358), chef (-0.281), seasoned (-0.228)
  Topic 2: cat (-0.409), dog (-0.409), puppy (-0.318), chased (-0.193), house (-0.193)

Explained variance ratio: [0.144 0.117 0.116]
Total explained variance: 0.377

Dominant topic per document:
  [pets   ] topic 2: My pet cat and dog play together in the garden every morning...
  [pets   ] topic 2: The puppy barked at the kitten while the cat chased the dog ...
  [pets   ] topic 2: A vet treats sick pets, from a coughing puppy to a limping c...
  [pets   ] topic 2: Cats purr and dogs wag their tails when their owners pet the...
  [cooking] topic 1: The chef baked bread in the oven and served it hot with butt...
  [cooking] topic 1: This recipe simmers garlic, tomato and onion in a pan for pa...
  [cooking] topic 1: The chef chopped garlic and tomato for the sauce and seasone...
  [cooking] topic 1: Bake the cake in the oven, stir the batter, and taste the sa...
  [space  ] topic 0: The telescope showed a galaxy of stars beyond our planet...
  [space  ] topic 0: An astronaut in orbit watched the moon and stars from the ro...
  [space  ] topic 0: The planet and its moon orbit a distant star in the galaxy...
  [space  ] topic 0: A rocket carried the telescope into orbit to study stars and...
Theme 'pets': dominant topics [2, 2, 2, 2] -> 4/4 docs on topic 2
Theme 'cooking': dominant topics [1, 1, 1, 1] -> 4/4 docs on topic 1
Theme 'space': dominant topics [0, 0, 0, 0] -> 4/4 docs on topic 0
```

Each topic is a single clean theme (space, cooking, pets), and every
document's dominant topic matches its theme, 4/4 in all three themes.

## Tests

```
pytest tests/test_lsa_topic_model.py -q
```

15 tests: fitting/topics/transform shapes, finite weights, explained
variance, per-theme dominant-topic grouping, reconstruction error
decreasing with more topics, and validation errors (empty docs,
token-less docs, `n_topics` too large or < 1, non-string docs, use
before fit).
