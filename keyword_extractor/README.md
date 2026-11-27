# Keyword Extractor

RAKE-style keyword/keyphrase extraction from plain text. Pure Python +
numpy — no NLTK or spaCy.

## How it works

1. Text is split into candidate phrases at stop words and punctuation.
2. Each word is scored as degree / frequency: degree counts how often
   the word co-occurs with other words inside a phrase, frequency is
   how often the word appears. Words that appear in long, varied
   phrases score highest.
3. Each phrase is scored as the sum of its word scores, then phrases
   are ranked highest-first (ties broken alphabetically).

## Run

```bash
cd /home/hatch/workspace/daily-builds
python keyword_extractor/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_keyword_extractor.py -q
```

## Example output

```
Top keyphrases:
   24.00  networks use many layers
   23.50  deep neural networks use
   20.50  neural network requires large
   19.17  machine learning models learn
```
