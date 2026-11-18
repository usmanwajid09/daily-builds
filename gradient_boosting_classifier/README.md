# Gradient Boosting Classifier, from scratch

Gradient boosting for classification, built with numpy only — no
scikit-learn. Each stage fits small CART regression trees to the negative
gradient of the loss (logistic loss for binary problems, multinomial
deviance / softmax loss for multi-class) and adds them to the ensemble,
scaled by the learning rate. Includes staged predictions (accuracy after
each stage) and gain-based feature importances.

## Usage

```bash
pip install -r requirements.txt
python gradient_boosting_classifier/demo.py
```

This fits 50 depth-3 trees on two interleaved synthetic crescents and
prints test accuracy, the accuracy after stages 1, 5, 10, 25 and 50, and
the feature importances.

## Run tests

```bash
python -m pytest tests/test_gradient_boosting_classifier.py -v
```
