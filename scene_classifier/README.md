# scene_classifier

A simple scene classifier built from scratch with **numpy only** — no
scikit-learn, no OpenCV, no image files.

It turns an image into a normalised RGB colour histogram
(`extract_features`, 8 bins per channel → a 24-dim vector) and classifies
it with k-nearest neighbours (`SceneClassifier`, majority vote, with
`predict_proba` returning the vote fractions per class).

Because there are no data files, the scenes are **generated in code**
(`generate_scene`): beach (blue sky + sea + sand), forest (green canopy +
brown trunk + ground), and city (grey buildings + sky + road), with a
little seeded noise so every seed gives a slightly different image.

## How to run

From the repo root (`daily-builds`):

```bash
/home/hatch/workspace/db-venv/bin/python scene_classifier/demo.py
```

## Demo output

```
Training set: 45 images (24-dim histogram features)
Classifier: k=3, classes=[np.str_('beach'), np.str_('city'), np.str_('forest')]
Test accuracy: 100.00% (15/15 correct)
  beach : 100.00% (5/5 correct)
  forest: 100.00% (5/5 correct)
  city  : 100.00% (5/5 correct)
Example predictions:
  true=beach  predicted=beach  votes={'beach': 1.0, 'city': 0.0, 'forest': 0.0}
  true=forest predicted=forest votes={'beach': 0.0, 'city': 0.0, 'forest': 1.0}
  true=city   predicted=city   votes={'beach': 0.0, 'city': 1.0, 'forest': 0.0}
All-blue image -> beach
```

(Exact numbers depend on the fixed seed 42 used in the demo; the classes
are so colour-separated that the synthetic test set scores 100%.)

## API

```python
from scene_classifier.scene_classifier import (
    extract_features, generate_scene, SceneClassifier,
)

img = generate_scene("beach", seed=0)          # (64, 64, 3) float in [0, 1]
feats = extract_features(img, bins=8)          # (24,) sums to 1

clf = SceneClassifier(k=3).fit(X_train, y_train)
clf.predict(X_test)                            # labels
clf.predict_proba(X_test)                      # vote fractions, rows sum to 1
clf.score(X_test, y_test)                      # accuracy
```

`SceneClassifier.fit` also accepts raw images (a 4D array) and converts
them with `extract_features` automatically. `train_classifier(X, y, k=3)`
and `predict(classifier, X)` are convenience wrappers.

## Validation

Bad input raises `ValueError` with a clear message: images must be 3D RGB
arrays with values in [0, 1] (float) or [0, 255] (uint8); `bins >= 1`;
`k >= 1` and `k <=` number of training samples; `predict` before `fit`;
unknown scene kind in `generate_scene`.

## Tests

```bash
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_scene_classifier.py -q
```

15 tests covering feature extraction, synthetic scene generation,
classification accuracy, probability outputs, and input-validation errors.
