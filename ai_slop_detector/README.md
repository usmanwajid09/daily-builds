# AI Slop Detector

A small, from-scratch (numpy-only) stylometric classifier that guesses
whether a short piece of text was written by a human or by a default-style
LLM. This is milestone 1 of the arc: dataset + features + a trained
classifier. See `ARC_QUEUE.md` at the repo root for what milestone 2 adds.

## Why "stylometric" and not "semantic"

This detector does not understand meaning. It looks at surface writing
style: sentence-length rhythm, contraction use, hedge/buzzword phrases,
vocabulary diversity, punctuation patterns, etc. -- the same category of
signal a careful human editor would notice ("this reads like it was
written by ChatGPT") without needing to fact-check the content.

## What's in this milestone

- `data/build_dataset.py` -- generates the labeled dataset (see
  "About the dataset" below) and writes `data/samples.csv`
  (200 rows: 100 `human`, 100 `ai`).
- `features.py` -- `extract_features(text)` turns raw text into an
  11-dimensional named feature vector (`FEATURE_NAMES`):
  - `avg_sentence_len_words`, `sentence_len_stdev` (uniform vs. bursty
    sentence rhythm)
  - `avg_word_len_chars`, `avg_syllables_per_word` (a rough proxy for
    reading complexity/formality)
  - `type_token_ratio` (vocabulary diversity)
  - `contraction_ratio_per_100w`, `first_person_ratio_per_100w`,
    `filler_slang_ratio_per_100w` (informal-writing tells)
  - `ai_marker_ratio_per_100w` (hedge/transition/buzzword phrases like
    "furthermore", "delve", "leverage", "seamless", "in conclusion")
  - `exclaim_question_ratio`, `comma_semicolon_ratio_per_sentence`
    (punctuation rhythm)
- `classifier.py` -- `StandardScaler` + `LogisticRegression`, both
  implemented from scratch in numpy (full-batch gradient descent, L2
  regularization, clipped sigmoid/log to avoid overflow), plus
  `confusion_matrix` / `classification_metrics`.
- `train.py` -- CLI that loads `samples.csv`, extracts features, does a
  stratified train/test split, trains the classifier, prints
  accuracy/precision/recall/F1 + confusion matrix for both splits, prints
  feature weights sorted by influence, and saves the trained weights +
  scaler stats to `trained_model.json`.
- `tests/` -- 26 tests covering feature extraction edge cases (empty
  text, punctuation-only text, single-sentence text), the scaler/model
  (constant-column guard, shape/label validation, convergence), and
  dataset integrity (balance, no duplicates, determinism).

## Usage

```bash
pip install -r ../requirements.txt   # numpy
python -m ai_slop_detector.data.build_dataset   # (re)generate samples.csv
python -m ai_slop_detector.train                # train + evaluate
python -m ai_slop_detector.train --test-size 0.3 --epochs 3000 --lr 0.05
python -m pytest ai_slop_detector/tests/
```

## About the dataset -- read this before trusting the results

This sandbox has no network access to download a real scraped AI-vs-human
corpus (e.g. HC3) or to call a live LLM API, so `data/samples.csv` is
**synthetically constructed from templates**, not scraped real-world text:

- The "human" generator writes informal, first-person, contraction-heavy
  paragraphs with uneven sentence length and mild tangents/slang, sampled
  from banks of opener/middle/closer sentence templates across 20 everyday
  topics (a weekend trip, a recipe, a movie, etc.).
- The "ai" generator writes formal, hedge-and-buzzword-heavy paragraphs in
  the well-known default LLM style ("Furthermore...", "it is important to
  note...", "leverage a seamless, holistic..."), with more uniform sentence
  length and a generic summarizing closer.
- Both are randomized (topic, sentence choice, order) with a fixed seed
  (`SEED = 20260706` in `build_dataset.py`) so the CSV is reproducible, but
  every sample is still built from a fairly small phrase bank.

**This means the label is true by construction (no labeling noise, unlike
scraped data) but the classes are almost certainly easier to separate than
real human vs. real modern LLM text.** On this dataset, the model reaches
100% train and 100% test accuracy -- that is a signal the *dataset* is
too easy / too lexically distinct (the `ai_marker_ratio_per_100w` feature
alone is close to a giveaway here), not evidence that this approach would
hit 100% on real-world text. See `REVIEW.md` for the self-review note on
this and what milestone 2 should do about it.

## Limitations (honest, not hedged)

- **Not a real detector.** Modern LLMs (especially with a system prompt
  discouraging "AI-isms") don't reliably produce these tells, and plenty
  of human writing (corporate emails, academic prose, non-native English)
  shares them. A `false positive` here means flagging a human as AI; a
  `false negative` means missing real AI text -- both are easy on a real
  corpus and this milestone doesn't attempt to bound that.
  Treat this as a demonstration of the *technique* (stylometric feature
  engineering + logistic regression from scratch), not a usable tool.
- **Synthetic dataset, not scraped data** -- see above. 200 samples across
  20 topics is small; a real project would need thousands of examples from
  real sources.
- **English-only**, and only prose paragraphs (no code, no lists, no
  dialogue formatting), whereas real "AI slop" complaints often come from
  bullet-point-heavy or emoji-heavy generated text this project doesn't
  model at all.
- **No cross-validation** -- a single stratified train/test split, not
  k-fold, so the reported metrics have higher variance than they'd
  otherwise appear to have (though on this dataset both splits currently
  hit 100%, so it isn't visible yet).
