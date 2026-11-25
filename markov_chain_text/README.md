# Markov Chain Text Generator

An order-N **word-level Markov chain** text generator, implemented with
the **standard library only**.

## How it works

Training slides a window of `order` words (default 2) over the text and
records, for every word sequence, which words can follow it. Generation
starts from a seed state (the first state of the training text, or
user-supplied seed words) and repeatedly picks a random recorded
successor, so the output mimics the style and vocabulary of the training
text. If a walk hits a dead end (a state with no recorded successor),
it restarts from the beginning state.

Generation is **deterministic** when a seed is given — either through
the constructor (`random_state=42`) or per call
(`generate(..., random_state=42)`).

## Usage

```python
from markov_chain_text import MarkovChainText

chain = MarkovChainText(order=2, random_state=42)
chain.fit("the cat sat on the mat the cat sat by the door")
print(chain.generate(length=12, seed_words="the cat"))
# "the cat sat on the mat the cat sat by the door" (or similar)

print(chain.n_states)  # distinct states learned
```

Fitting empty text, text too short to learn a transition (at most
`order` words), or generating with the wrong number of seed words / seed
words that never occur in the training text, raises `ValueError`/
`TypeError`/`RuntimeError` with a clear message.

## Run the demo

```bash
python markov_chain_text/demo.py
```

Trains on a short paragraph about machine learning and generates a
30-word passage from the seed words "machine learning" — the same text
on every run.

## Run the tests

```bash
pytest tests/test_markov_chain_text.py -q
```
