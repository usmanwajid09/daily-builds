# Readability Scorer

Readability metrics for English text: Flesch Reading Ease,
Flesch-Kincaid Grade Level, Gunning Fog Index and SMOG (approximate).
Pure Python + numpy.

## How it works

- Sentences are split on `.` `!` `?`; words and syllables are counted
  (syllables use the standard vowel-group heuristic with a silent-e
  rule, plus a small exception list).
- Flesch Reading Ease = 206.835 − 1.015 × words/sentence − 84.6 ×
  syllables/word (0–100, higher = easier).
- Flesch-Kincaid Grade = 0.39 × words/sentence + 11.8 ×
  syllables/word − 15.59 (US school grade).
- Gunning Fog = 0.4 × (words/sentence + % of words with 3+ syllables).
- SMOG = 1.043 × √(complex words × 30 / sentences) + 3.1291; needs at
  least 3 sentences, otherwise `smog` is `None`.

## Run

```bash
cd /home/hatch/workspace/daily-builds
python readability_scorer/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_readability_scorer.py -q
```

## Example output

```
Simple text:
  Words: 26, sentences: 5, syllables: 26
  Flesch Reading Ease: 100.0 (Very Easy)
  Flesch-Kincaid Grade: 0.0
  Gunning Fog: 2.08
  SMOG: 3.13
```
