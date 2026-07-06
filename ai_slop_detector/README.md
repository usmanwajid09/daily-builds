# AI Slop Detector

A small, from-scratch (numpy-only) stylometric classifier that guesses
whether a short piece of text was written by a human or by a default-style
LLM. Arc complete as of milestone 2: dataset + features + classifier
(milestone 1), rigorous evaluation + a CLI demo (milestone 2). See
`ARC_QUEUE.md` at the repo root for the full milestone list.

## Why "stylometric" and not "semantic"

This detector does not understand meaning. It looks at surface writing
style: sentence-length rhythm, contraction use, hedge/buzzword phrases,
vocabulary diversity, punctuation patterns, etc. -- the same category of
signal a careful human editor would notice ("this reads like it was
written by ChatGPT") without needing to fact-check the content.

## Headline result -- read this first

Cross-validated on the synthetic template dataset, this model gets
**100% mean accuracy across 5 folds**. Evaluated on 20 hand-written,
out-of-distribution "hard" sentences designed to violate the dataset's
assumptions (formal-but-human writing, casual-but-AI writing), it gets
**25% accuracy** -- worse than always guessing one class. Both numbers are
real and reproducible (`python -m ai_slop_detector.evaluate`). Read them
together, not separately: this project demonstrates the *technique*
(stylometric feature engineering + logistic regression from scratch,
plus honest evaluation practice) rather than a working detector. See
"Evaluation" and "Limitations" below for what the gap means and why it
wasn't fixed by making the dataset easier.

## What's in this project

- `data/build_dataset.py` -- generates the main labeled training dataset
  and writes `data/samples.csv` (200 rows: 100 `human`, 100 `ai`; see
  "About the dataset" below).
- `data/hard_examples.csv` -- 20 hand-written, out-of-distribution
  sentences (10 human, 10 ai) built specifically to violate the main
  dataset's stylistic assumptions, each with a `note` explaining why it's
  hard. Never used for training, only for evaluation.
- `features.py` -- `extract_features(text)` turns raw text into an
  11-dimensional named feature vector (`FEATURE_NAMES`): sentence-length
  rhythm (`avg_sentence_len_words`, `sentence_len_stdev`), word/syllable
  complexity (`avg_word_len_chars`, `avg_syllables_per_word`), vocabulary
  diversity (`type_token_ratio`), informal-writing tells
  (`contraction_ratio_per_100w`, `first_person_ratio_per_100w`,
  `filler_slang_ratio_per_100w`), buzzword/hedge density
  (`ai_marker_ratio_per_100w`), and punctuation rhythm
  (`exclaim_question_ratio`, `comma_semicolon_ratio_per_sentence`).
- `classifier.py` -- `StandardScaler` + `LogisticRegression` from scratch
  in numpy (full-batch gradient descent, L2 regularization, clipped
  sigmoid/log), plus `confusion_matrix`, `classification_metrics`
  (single positive-class view), and `classification_report` /
  `format_classification_report` (per-class precision/recall/F1/support +
  macro average -- added in milestone 2).
- `dataset_utils.py` -- shared dataset loading, label validation,
  stratified train/test split, and stratified k-fold split, used by both
  `train.py` and `evaluate.py` (added in milestone 2, refactored out of
  `train.py` to avoid duplicating the same logic in two places).
- `train.py` -- CLI: single stratified train/test split, trains the
  classifier, prints metrics for both splits and feature weights sorted
  by influence, saves `trained_model.json`.
- `evaluate.py` -- CLI (milestone 2): stratified k-fold cross-validation
  on `samples.csv` (mean/stdev accuracy and macro-F1 across folds) plus
  the hard-example check described above, with a per-example
  right/wrong breakdown.
- `demo.py` -- CLI (milestone 2): classify one string, a file of lines, or
  run an interactive REPL; prints the prediction, `p_ai`, and the top
  contributing features (`feature_value * learned_weight`) so the
  prediction is explainable, not just a bare label.
- `tests/` -- 43 tests: feature extraction edge cases, the scaler/model,
  dataset integrity (`samples.csv` and `hard_examples.csv`), the k-fold
  splitter, and the classification report.

## Usage

```bash
pip install -r ../requirements.txt   # numpy

# (re)generate the training dataset
python -m ai_slop_detector.data.build_dataset

# train on a single split, print metrics + feature weights, save the model
python -m ai_slop_detector.train
python -m ai_slop_detector.train --test-size 0.3 --epochs 3000 --lr 0.05

# rigorous evaluation: k-fold CV + the hard-example check
python -m ai_slop_detector.evaluate
python -m ai_slop_detector.evaluate --folds 10

# classify text interactively, or one string, or a file of lines
python -m ai_slop_detector.demo
python -m ai_slop_detector.demo "Furthermore, this seamless approach can help unlock lasting value."
python -m ai_slop_detector.demo --file some_notes.txt

python -m pytest ai_slop_detector/tests/
```

## About the dataset -- read this before trusting the cross-validation number

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

The label is true by construction (no labeling noise, unlike scraped data)
but the two template families differ on nearly every feature at once, so
they're trivially separable -- milestone 1's self-review verified this by
ablation (removing the single most obvious feature, or using only that
feature, both still gave 100%). That's *why* milestone 2 added
`hard_examples.csv` and `evaluate.py` instead of just reporting the
5-fold CV number: cross-validation alone would still say "100%, ship it,"
which would be actively misleading.

## Evaluation

`python -m ai_slop_detector.evaluate` runs two checks:

1. **5-fold cross-validation on `samples.csv`.** Mean accuracy 100%,
   stdev 0.000 -- every fold separates perfectly, consistent with the
   ablation finding above. This measures "can a linear model separate the
   two template families," not "can this detect real AI text."
2. **The 20-example hard set** (`hard_examples.csv`), trained on all of
   `samples.csv` and evaluated only on these untouched examples: **25%
   accuracy**, human-class precision 0.31/recall 0.40, ai-class precision
   0.14/recall 0.10. Worse than a coin flip on this small set. The
   per-example breakdown (`evaluate.py`'s output) shows a clear pattern:
   the model reliably mistakes formal-but-human writing (corporate email,
   textbook prose, a LinkedIn congratulations post) for AI, and reliably
   mistakes casual-but-AI writing (contractions, slang, first person, no
   buzzwords) for human. In other words, it learned "formal register with
   buzzwords = ai, informal register with contractions = human" -- which
   is true of *this dataset's templates* and not reliably true of real
   writing in either direction.

This gap is the honest headline result of the project: the technique
(stylometric features + logistic regression + rigorous eval methodology)
works as implemented, but the training data's stylistic assumptions don't
transfer to text that violates them, which is most real text.

## Limitations (honest, not hedged)

- **Not a real detector, and milestone 2's own evaluation demonstrates
  why**, not just asserts it: see "Evaluation" above. A `false positive`
  here means flagging a human as AI; a `false negative` means missing
  real AI text -- the hard-example check shows both happen constantly the
  moment text doesn't match the training templates' register.
- **Synthetic dataset, not scraped data.** 200 training samples across 20
  topics, plus 20 hand-written hard examples, is small either way; a real
  project would need thousands of examples from real, messy sources, and
  ideally text generated by several different LLMs and prompting styles
  (a system prompt telling an LLM to "write casually" defeats every
  feature in this project).
- **English-only**, and only prose paragraphs (no code, no lists, no
  dialogue formatting, no emoji), whereas real "AI slop" complaints often
  come from bullet-point-heavy or emoji-heavy generated text this project
  doesn't model at all.
- **The hard-example set is itself small and hand-picked (n=20)**, so its
  25% shouldn't be over-read as a precise real-world error rate either --
  treat it as "clearly and substantially worse than the training-set
  number," not as a calibrated estimate.
- **Fixing this properly is out of scope for this arc.** It would need
  real data (blocked: no network access to a corpus or a live LLM API
  from this sandbox) rather than a better classifier -- the from-scratch
  logistic regression and feature extraction aren't the bottleneck here,
  the training data's narrowness is.
