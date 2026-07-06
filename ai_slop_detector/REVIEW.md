# ai-slop-detector — self-review log

Running log, appended per milestone. Most recent milestone first.

## Milestone 1 (2026-07-06) — dataset + stylometric features + logistic regression classifier

### Real bug found and fixed
`features.py`'s original contraction regex, `\b[a-zA-Z]+'(?:t|re|ve|ll|d|m|s)\b`,
treated a bare `'s` as always being a contraction ("it's" = "it is"). That's
wrong: `'s` is genuinely ambiguous in English between the is/has contraction
and a possessive ("today's forecast", "the neighbor's garden"). Grepping the
committed dataset found 24 occurrences of `today's` (from the AI-class
opener template "In today's fast-paced world...") and 15 of `neighbor's`
(from the "my neighbor's garden" topic, which appears in both classes) that
were being miscounted as contractions -- i.e. inflating a feature whose
whole point is "informal human writing uses more contractions" with hits
coming from a formal AI-labeled template. Fixed by splitting the regex:
`'t/'re/'ve/'ll/'d/'m` stay unconditionally unambiguous, and a bare `'s` is
only counted when it follows a fixed set of words that are never used
possessively in English (it/that/there/here/what/who/how/when/where/why/
let/he/she) -- so "it's" and "that's" still count, but "today's" and
"neighbor's" no longer do. Added 3 regression tests
(`test_possessive_s_is_not_counted_as_a_contraction`,
`test_unambiguous_s_contractions_are_still_counted`,
`test_mixed_possessive_and_contraction_counts_only_the_contraction`).
Effect after the fix: `contraction_ratio_per_100w`'s learned weight moved
from -0.960 to -1.015 (a cleaner, slightly stronger human-ward signal, as
expected once the possessive noise was removed) with no change in
train/test accuracy (both were already 100% -- see next finding for why
that number should not be read as "the detector is 100% accurate").

### Not a bug, but the headline number is misleading -- quantified, not just asserted
Both train and test accuracy are 100%. Before shipping that number I
checked *why*, since a perfect score on a hand-built dataset is a red flag
for "the dataset is trivially separable," not "the model/features are
great":
- Retrained with `ai_marker_ratio_per_100w` (the most obviously
  giveaway-y feature -- literal buzzword counting) removed entirely: test
  accuracy stayed at 100%.
- Retrained using *only* `ai_marker_ratio_per_100w` and nothing else: also
  100%.
- This means the two template families differ on essentially every axis
  simultaneously (sentence-length uniformity, contraction use, first-person
  pronoun use, vocabulary diversity, buzzwords) by construction, so almost
  any subset of these features alone reaches perfect separation. The
  reported 100%/100% is real (not a leakage bug -- scaler is fit on train
  only, split is stratified and non-overlapping, `test_no_duplicate_texts_
  across_dataset` passes) but it measures "can a linear model separate two
  rigidly-templated phrase banks," not "can this detect real human vs. real
  LLM text."
- Probed the trained model on 3 hand-written sentences *outside* the
  template banks to sanity-check generalization direction (not part of the
  automated test suite -- an exploratory check, documented here instead):
  - `"Honestly furthermore I don't think we're ready, tbh, but let's
    leverage what we've got."` (deliberately mixes AI buzzwords into an
    informal, contraction-heavy, one-sentence structure) -> predicted
    **human** at p(ai)=0.003. This is a real weakness: the model is
    picking up more on paragraph-level *structure* (sentence count,
    average length) learned from multi-sentence training examples than on
    buzzwords alone, so a short, single-sentence mixed-signal input isn't
    handled well.
  - `"The quarterly report indicates a decline in revenue across all
    regions."` (formal, no contractions, no buzzwords, no first person) ->
    p(ai)=0.473, i.e. genuinely near the decision boundary rather than
    confidently on either side. Arguably the *correct* behavior (this
    sentence is ambiguous), but it's a coincidence of this input rather
    than evidence the model handles ambiguous cases well in general.
  - A casual first-person sentence with a contraction predicted human with
    high confidence, as expected.
- **Not fixed in this milestone** -- fixing dataset-triviality properly
  needs either (a) real scraped text (blocked: no network access to a
  corpus or a live LLM API from this sandbox) or (b) deliberately
  constructing harder, single-sentence and mixed-signal examples in the
  template generator. Flagging this explicitly as the most important
  input to milestone 2's evaluation section rather than silently
  reporting "100% accuracy" as if it were a finished, trustworthy result.
  README's "About the dataset" and "Limitations" sections state this
  plainly for anyone reading the repo without this file.

### Reviewed and judged not worth changing this milestone
- `avg_syllables_per_word` uses a vowel-group heuristic (count runs of
  a/e/i/o/u/y, subtract 1 for a trailing silent "e"). It's wrong on
  plenty of individual words (e.g. "the" scores 1, correct; "queue" scores
  1, correct by luck; genuinely irregular words like "colonel" will be off)
  but it's used only as an *averaged* feature across a whole paragraph, so
  per-word noise mostly cancels out. A real syllable dictionary (CMUdict)
  isn't available offline here; documented as a known heuristic rather than
  silently presented as exact.
- `AI_MARKER_PHRASES` substring-matches against the whole lowercased text
  rather than tokenizing first, so a marker phrase that happened to span a
  sentence boundary (e.g. "...end. Overall, ...") would still match
  correctly since it's checked as a raw substring -- confirmed this is
  intentional and correct, not a bug.
- Checked `git ls-files` before each commit; `trained_model.json` (a
  regenerable training artifact, not source) is gitignored rather than
  committed, matching how other arcs in this repo don't commit generated
  model/report artifacts.

## Milestone 2 (2026-07-06) — rigorous evaluation, hard-example set, and a CLI demo

### Real bug found and fixed
`evaluate.py --folds 1` (or 0, or negative) crashed with a multi-frame
`ValueError` traceback originating deep in `dataset_utils.stratified_k_folds`
rather than a clean CLI error naming the bad flag. `stratified_k_folds`
correctly validates `k >= 2`, but argparse doesn't know that constraint, so
the exception surfaced as an unhandled stack trace pointing at the wrong
file. Fixed by validating `args.folds` in `evaluate.main()` and calling
`parser.error(...)` (argparse's standard mechanism -- prints a proper
usage line + message to stderr and exits 2) before any dataset loading
happens. Added a subprocess-based regression test
(`test_cli_rejects_folds_less_than_2_with_clean_error`) asserting no
"Traceback" appears in stderr and the friendly message does.

### The headline finding this milestone was built to produce
Milestone 1's self-review flagged, but didn't fix, that 100%/100% train/test
accuracy on `samples.csv` reflects a trivially-separable synthetic dataset.
Milestone 2's job was to actually quantify how bad the real gap is, not
just gesture at it. Built `data/hard_examples.csv` (20 hand-written
sentences deliberately violating the training templates' assumptions --
formal/buzzwordy-but-human, casual/contraction-heavy-but-ai) and
`evaluate.py` (5-fold stratified CV + a held-out hard-example check).
Result: **5-fold CV mean accuracy 100% (stdev 0.000) vs. 25% accuracy on
the 20 hard examples** -- worse than chance. The per-example breakdown
shows a specific, legible failure mode rather than random noise: every
miss is explained by "formal/buzzword register -> predicted ai" or
"casual/contraction register -> predicted human," regardless of the true
label. This confirms the exact mechanism milestone 1 flagged: the model
learned the two template families' registers, not "AI-ness" vs.
"human-ness." Not fixed (would require real training data, which needs
network access this sandbox doesn't have) -- the point of this milestone
was to make the gap impossible to miss rather than to close it, and that's
now baked into the README's headline section rather than buried in
REVIEW.md where a reader would have to go looking for it.

### Reviewed and judged not worth changing this milestone
- `stratified_k_folds`'s round-robin bucket assignment can, for a
  pathologically small class relative to `k` (e.g. `--folds 50` on a
  10-example class), produce folds where that class has 0 test examples.
  `classification_report` already handles zero-support classes without
  crashing (returns 0.0 rather than dividing by zero, tested in
  `test_classification_report_handles_a_class_with_zero_support`), so this
  degrades gracefully rather than crashing; not worth adding an artificial
  cap on `--folds` for a 200-example dataset where the default (5) and any
  reasonable value are fine.
- `demo.py` auto-trains a model via `subprocess.run(["-m",
  "ai_slop_detector.train"])` if `trained_model.json` is missing. This
  assumes the demo is invoked from the repo root (so `python -m
  ai_slop_detector.X` resolves) -- true for every other CLI in this
  project already, so not a new assumption, just inherited.
- Confirmed no train/hard-example leakage: `run_hard_example_check` fits
  the scaler and model on all of `samples.csv` and only ever calls
  `.transform()` (never `.fit()`) on the hard-example features; the two
  CSVs share zero rows (checked manually -- `hard_examples.csv` isn't
  generated from `build_dataset.py`'s template banks at all, it's
  independently hand-written).
- `git ls-files` checked before every commit this milestone too;
  `trained_model.json` regenerated by `train.py`/`demo.py` stayed
  gitignored (see milestone 1's REVIEW.md entry).
