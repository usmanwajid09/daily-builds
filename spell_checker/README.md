# spell_checker

Edit-distance spell checker, from scratch (Norvig-style), using only the
standard library.

## What it does

- Builds a vocabulary with word frequencies from a small built-in corpus
  (or a custom corpus: a text string, or a `{word: count}` dict).
- To correct a word it generates all candidates one edit away — deletes,
  transposes, replaces, inserts — and, if none is in the vocabulary,
  candidates two edits away. Candidates are ranked by corpus frequency,
  so the most common plausible word wins.
- `correct(word)` returns the best correction, or the word unchanged if
  it is already in the vocabulary or has no close candidate.
- `correct_sentence(text)` corrects every word while keeping punctuation,
  spacing and the capitalisation of each word's first letter.

## Usage

```python
from spell_checker import SpellChecker

checker = SpellChecker()
checker.correct("speling")    # 'spelling'
checker.correct("recieve")    # 'receive'
checker.correct("teh")        # 'the'
checker.correct("spelling")   # 'spelling' (already known: unchanged)
checker.correct_sentence("Teh students practise speling.")
# 'The students practise spelling.'

# Custom corpus: plain text or a word -> count mapping.
checker = SpellChecker(corpus="apple banana apple cherry apple")
checker.correct("appl")       # 'apple'
checker.frequency("apple")     # 3
```

## Demo

Run from the repo root:

```
python spell_checker/demo.py
```

Output summary: vocabulary of 171 words from 365 corpus tokens
(`the` is the most common word, 81 occurrences). Corrections shown:
`speling -> spelling`, `recieve -> receive`, `teh -> the`,
`wrting -> writing`, `adress -> address`, `seperate -> separate`.
The sentence `Teh students practise speling and hope to recieve good
news.` becomes `The students practise spelling and hope to receive
good news.` Known words stay unchanged and gibberish with no close
match (e.g. `zxqv`) is left as-is.

## Tests

```
python -m pytest tests/test_spell_checker.py -q
```

25 tests: the three demo corrections, known-word passthrough, empty /
non-string / non-alphabetic input validation, capitalisation,
punctuation and numbers in sentences, frequency ranking with a custom
corpus, `max_edit_distance` validation, and edit-distance-1 transposes.
