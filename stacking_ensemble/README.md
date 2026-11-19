# Stacking Ensemble

A stacking classifier built from scratch with numpy only.

## What it does

Several diverse base classifiers are trained, and their out-of-fold
predictions on the training data are used as features to train a meta
learner, which makes the final prediction. Out-of-fold predictions mean
the meta learner never sees predictions a base learner made on data it
was trained on, which avoids leaking the training labels.

The module also ships two tiny built-in learners so it works with no
other dependency:

- `SimpleLogisticRegression` — binary logistic regression (gradient descent)
- `DecisionStump` — a one-level decision tree

## How to run

```bash
python stacking_ensemble/demo.py
```

## Demo result

Two synthetic 2D clusters (100 points per class, 80/20 train/test split).
Base learners: logistic regression + decision stump; meta learner:
logistic regression, 5-fold out-of-fold predictions.

- Base learner test accuracies: logistic regression 1.000, stump 0.925
- Stacking test accuracy: **0.925** (on this easy, well-separated data the
  stack matches the stump and trails the best base learner — stacking
  helps most when the base learners disagree more)
