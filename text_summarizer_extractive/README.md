# Extractive Text Summarizer

Ranks the sentences of a document by term frequency and keeps the
top-N, in their original order. No training data, no external models:
content words (stopwords removed) are scored by how often they appear
in the document, each sentence's score is the average term-frequency
score of its content words, and the highest-scoring sentences win.

## Usage

```python
from text_summarizer_extractive import summarize

text = (
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy. "
    "Plants capture sunlight with chlorophyll in their leaves. "
    "During photosynthesis, plants absorb carbon dioxide from the air and water from the soil."
)
print(summarize(text, n_sentences=2))
# ['Photosynthesis is the process by which green plants convert sunlight into chemical energy.',
#  'Plants capture sunlight with chlorophyll in their leaves.']
```

`summarize(text, n_sentences=2)` returns a `list[str]` of the selected
sentences in the order they appear in the text. `n_sentences` must be
>= 1 and no larger than the number of sentences; empty text raises
`ValueError`. Sentence splitting handles `.`, `!` and `?` terminators.

There is also a class API:

```python
from text_summarizer_extractive import ExtractiveSummarizer

summarizer = ExtractiveSummarizer(n_sentences=2)
summary = summarizer.summarize(text)
scores = summarizer.scores(text)  # per-sentence scores, original order
```

Helper functions (`split_sentences`, `tokenize`, `content_words`,
`word_frequencies`, `score_sentences`) are exported from the package
for inspection and testing.

## Demo

Run `python text_summarizer_extractive/demo.py` from the repo root.
It summarizes an 8-sentence paragraph about photosynthesis down to
2 sentences and prints every sentence's score:

```
Original paragraph: 8 sentences
  [1] score=0.0385  Photosynthesis is the process by which green plants convert sunlight into chemical energy.
  [2] score=0.0338  Plants capture sunlight with chlorophyll in their leaves.
  ...
Summary (2 sentences, original order kept):
  - Photosynthesis is the process by which green plants convert sunlight into chemical energy.
  - Plants capture sunlight with chlorophyll in their leaves.
Key terms checked: ['photosynthesis', 'plants', 'sunlight']
Key terms present in summary: ['photosynthesis', 'plants', 'sunlight']
All key terms present in the summary.
```

The two chosen sentences are the highest-scoring ones because they
pack the document's most frequent content words — *photosynthesis*,
*plants* and *sunlight* — into few words.

## Tests

`pytest tests/test_text_summarizer_extractive.py -q` — 26 tests covering
validation errors, edge cases, scoring behaviour, the class API and
the demo paragraph.
