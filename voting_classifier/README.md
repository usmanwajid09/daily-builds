# Voting Classifier

A hard/soft voting ensemble classifier built from scratch with numpy only.

## What it does

Combines several base classifiers into one:

- **Hard voting**: each base learner casts one vote (optionally
  weighted); the class with the most votes wins.
- **Soft voting**: the predicted class probabilities of the base
  learners are averaged (optionally weighted); the class with the
  highest average probability wins.

The module also ships two tiny built-in learners so it works with no
other dependency:

- `SimpleLogisticRegression` — binary logistic regression (gradient descent)
- `SimpleKNNClassifier` — k-nearest neighbours (Euclidean distance)

## How to run

```bash
python voting_classifier/demo.py
```

## Demo result

Two synthetic 2D clusters (100 points per class, 80/20 train/test split).
Base learners: logistic regression + k-NN (k=5).

- Base learner test accuracies: logistic regression 1.000, k-NN 1.000
- Soft voting test accuracy: **1.000**
- Hard voting test accuracy: **1.000**
