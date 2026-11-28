# logreg_text_classifier

A multiclass logistic-regression text classifier over TF-IDF features,
implemented with **numpy only** — no scikit-learn, no extra dependencies.

## How it works

1. **Tokeniser** — lowercases text and extracts alphanumeric words
   (`[a-z0-9]+`).
2. **TF-IDF vectoriser** (`TfidfVectorizer`, built in) — vocabulary from
   the training documents only; sublinear term frequency `1 + log(count)`,
   smoothed IDF `log((1 + N) / (1 + df)) + 1`, and L2-normalised rows.
   Documents made only of out-of-vocabulary words become all-zero rows
   and are still classified (via the learned bias).
3. **Logistic regression** (`LogisticRegressionTextClassifier`) — softmax
   + cross-entropy loss with L2 regularisation, trained by batch gradient
   descent. Configurable `learning_rate`, `epochs`, `reg`, and
   `random_state` (seeded weight initialisation, so fits are
   deterministic). `fit` returns `self`; `predict` returns labels;
   `predict_proba` returns rows that sum to 1.

Input validation raises clear errors: `ValueError` for empty training
data, mismatched `X`/`y` lengths, or documents with no tokens;
`TypeError` for non-string documents; `RuntimeError` for predicting
before fitting.

## Usage

```python
from logreg_text_classifier import LogisticRegressionTextClassifier

clf = LogisticRegressionTextClassifier(learning_rate=0.5, epochs=300, reg=0.01)
clf.fit(
    ["the striker scored a goal in the football match",
     "bake the bread in the oven until golden"],
    ["sports", "cooking"],
)
print(clf.predict(["the goalkeeper saved the penalty"]))
# ['sports']
print(clf.predict_proba(["the goalkeeper saved the penalty"]))
# [[...]]  # one row, sums to 1
```

## Demo

Run from the repo root:

```bash
python logreg_text_classifier/demo.py
```

The demo trains on 24 short texts (8 each on sports, cooking, and
technology), holds out the last 2 documents per class as a test set,
and prints vocabulary size, train/test accuracy, and example
predictions with probabilities.

## Example output

```text
Logistic-regression text classifier (TF-IDF + softmax)
Classes: ['cooking', 'sports', 'technology']
Vocabulary size: 102
Train documents: 18  Test documents: 6
Train accuracy: 1.00
Test accuracy:  1.00

Example predictions:
  'the striker scored a goal in the final match'
    -> sports  {'cooking': 0.087, 'sports': 0.827, 'technology': 0.086}
  'bake the cake in the oven with sugar and eggs'
    -> cooking  {'cooking': 0.764, 'sports': 0.122, 'technology': 0.114}
  'the python code runs the algorithm on the server'
    -> technology  {'cooking': 0.085, 'sports': 0.103, 'technology': 0.812}
  'the goalkeeper saved the penalty in the stadium'
    -> sports  {'cooking': 0.143, 'sports': 0.712, 'technology': 0.144}
  'chop the onion and stir the garlic into the sauce'
    -> cooking  {'cooking': 0.742, 'sports': 0.139, 'technology': 0.118}
  'the neural network trains on the cloud computer'
    -> technology  {'cooking': 0.135, 'sports': 0.158, 'technology': 0.708}
```

## Tests

```bash
pytest tests/test_logreg_text_classifier.py -q
```

14 tests covering the fit/predict round-trip, demo-data accuracy above
chance, probability rows summing to 1 and staying in [0, 1],
unknown-word handling, validation errors, determinism, tokenisation,
and training-only vocabulary.
