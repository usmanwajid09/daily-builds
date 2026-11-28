# ngram_language_model

An n-gram language model (orders 1–3) with add-k smoothing and perplexity.
Standard library only — no third-party dependencies.

## How it works

Sentences are tokenised into lowercase alphanumeric words (punctuation is
dropped), padded with `<s>` start tokens and a `</s>` end token, and n-gram
counts are built for every order from 1 up to `order`.

Conditional probabilities use **add-k smoothing**: the count of
(word after context) plus `k` is divided by the total count of the context
plus `k` times the vocabulary size `V`:

    P(w | context) = (count(context, w) + k) / (count(context) + k * V)

Because of the smoothing, even words never seen after a context (or never
seen at all) get a small non-zero probability, so the probability of any
sentence is never zero and perplexity is always finite.

Perplexity is `exp(-mean log probability per token)`, counting one `</s>`
token per sentence. Lower is better: text like the training corpus scores
much lower than unrelated or garbled text.

Generation is greedy argmax by default (deterministic, ties broken
alphabetically), with optional random sampling via `sample=True`.

## Usage

```python
from ngram_language_model import NGramLanguageModel

corpus = [
    "the cat sits on the mat",
    "the dog runs in the park",
    "the cat chases the dog",
]

model = NGramLanguageModel(order=2, k=0.1)
model.fit(corpus)

print(model.cond_prob("cat", ("the",)))   # P(cat | the)
print(model.log_prob("the cat sits on the mat"))
print(model.perplexity(["the cat sits on the mat"]))
print(model.generate("the cat", max_len=10))
print(model.generate("the cat", max_len=10, sample=True, random_state=42))
```

## Run the demo

```bash
python ngram_language_model/demo.py
```

## Example output

```
Vocabulary size: 33
Total tokens: 60
Perplexity (in-domain 'the dog chases the cat'): 6.39
Perplexity (out-of-domain 'quantum banana spaceship zebra'): 43.61
Generated from 'my cat': my cat likes milk and fish
Generated from 'a happy': a happy dog wags its tail
Generated from 'the dog': the dog barks at the cat chases the cat chases the cat
```

## Tests

```bash
pytest tests/test_ngram_language_model.py -q
```

16 tests: conditional probabilities sum to 1, in-domain perplexity is lower
than out-of-domain, training-text perplexity is finite and positive,
generation starts with the seed and is deterministic, add-k smoothing gives
unseen words non-zero probability, validation errors (bad order, `k <= 0`,
empty corpus, use before fit), and order-1 / order-3 models work.
