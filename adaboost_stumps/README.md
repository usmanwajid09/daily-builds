# AdaBoost with Decision Stumps, from scratch

AdaBoost (Freund & Schapire, 1997) over one-level decision stumps,
implemented with numpy only — no scikit-learn. Each round finds the
stump with the lowest weighted error, gives it a vote proportional to
its accuracy, and up-weights the samples it got wrong so the next
stump focuses on the hard cases. Training stops early on a perfect
stump or one no better than chance.

`predict` uses all stumps; `staged_predict` returns the prediction
after each round, so you can watch accuracy improve as stumps are
added. `feature_importances_` sums each stump's vote by the feature it
splits on, normalised to 1.

## Usage

```bash
pip install -r requirements.txt
python -m adaboost_stumps.demo
```

This generates points from a 2D Gaussian with the label given by a
diagonal boundary (`x0 + x1 > 0.5`) that no single axis-aligned stump
can separate well, fits AdaBoost with up to 30 stumps, and prints test
accuracy, the number of stumps used, staged accuracy, and feature
importances.

## Run tests

```bash
python -m pytest tests/test_adaboost_stumps.py -v
```
