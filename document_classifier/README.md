# Document Classifier

Classifies synthetic document images as **invoice**, **letter**, **form**
or **receipt** from layout / texture features, using a nearest-centroid
classifier built from scratch with numpy + stdlib only.

## How it works

1. **Synthetic documents** — `make_document(doc_type, seed)` renders a
   200×150 grayscale page (255 = paper, 0 = ink) with a distinct layout
   per class, using only numpy drawing primitives:
   - *invoice*: header block + table grid (horizontal + vertical rules)
     with short text segments inside the cells, and a total line.
   - *letter*: sender block, dense full-width paragraph lines, signature line.
   - *form*: wide margins, labelled field boxes, checkboxes, signature box.
   - *receipt*: narrow 90 px column, many short item lines, dense total block.
2. **Features** — `extract_features(image)` returns a 14-D vector:
   ink density; horizontal-line count (rows > 50% ink); vertical-line
   count; left/right/top/bottom margins; text-line count; mean line
   length; ink density in the top/middle/bottom thirds; approximate
   enclosed-box count (intersections of long horizontal and vertical
   runs ÷ 4); mean horizontal ink-run length.
3. **Classifier** — nearest centroid: each class is summarised by the
   mean of its standardised training feature vectors, and a new document
   is assigned to the closest centroid (Euclidean distance).
   `predict_proba` turns negative distances into probabilities with a
   softmax.

## Usage

```python
from document_classifier import make_document, train_classifier

train_X = [make_document(cls, seed) for cls in ("invoice", "letter", "form", "receipt") for seed in range(20)]
train_y = [cls for cls in ("invoice", "letter", "form", "receipt") for _ in range(20)]

clf = train_classifier(train_X, train_y)
print(clf.predict(make_document("receipt", seed=42)))          # 'receipt'
print(clf.predict_proba(make_document("invoice", seed=4242)))  # {'invoice': 0.986, ...}
```

Input validation raises `ValueError` with a clear message for empty or
non-2-D images, unknown document types / labels, predicting before
fitting, and mismatched image/label counts.

## Demo

Run from the repo root:

```
python document_classifier/demo.py
```

Output (train 20 docs/class, test 20 held-out docs/class):

```
Document classifier — nearest centroid over layout features
Train: 20 docs/class  |  Test: 20 held-out docs/class
Overall accuracy: 100.0%

Per-class accuracy:
  invoice  100.0%  (20/20)
  letter   100.0%  (20/20)
  form     100.0%  (20/20)
  receipt  100.0%  (20/20)

Confusion matrix (rows = true, cols = predicted):
true \ pred   invoice   letter     form  receipt
invoice           20        0        0        0
letter             0       20        0        0
form               0        0       20        0
receipt            0        0        0       20

Example: predict_proba(invoice, seed=4242) ->
  invoice  0.986
  letter   0.011
  form     0.001
  receipt  0.001
```

## Tests

18 pytest tests in `tests/test_document_classifier.py` — generation
(shape, determinism, per-class layout differences), feature extraction
(blank page, invalid input), and the classifier (held-out accuracy
≥ 85%, predict/proba contract, predict-before-fit, invalid labels,
confusion-matrix counts, determinism). All 18 pass.
