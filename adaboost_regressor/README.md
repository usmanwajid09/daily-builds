# AdaBoost Regressor, from scratch

AdaBoost for regression, built with numpy only — no scikit-learn. An
ensemble of decision stumps is trained sequentially: each stump is fit to
the sample-weighted data, its error is measured with the linear loss
(absolute error scaled by the worst error in the stage, or the square
loss), and sample weights are boosted for the points it got wrong. The
final prediction is the weighted median of the stumps' predictions,
which is robust to a few badly wrong stumps.

## Usage

```bash
pip install -r requirements.txt
python adaboost_regressor/demo.py
```

This fits 100 boosted stumps on Friedman #1 synthetic data (a non-linear,
5-feature target) and prints the test R^2 of the full ensemble and of
the first 1, 10 and 50 stumps, plus the feature importances. A single
stump is a weak learner on this data (test R^2 ≈ 0.19); the boosted
ensemble reaches ≈ 0.54 — boosting only helps where the reweighted
stumps diversify, which is why the demo uses multi-feature data rather
than a smooth 1-D curve (where every stump picks the same split and
the ensemble plateaus).

## Run tests

```bash
python -m pytest tests/test_adaboost_regressor.py -v
```
