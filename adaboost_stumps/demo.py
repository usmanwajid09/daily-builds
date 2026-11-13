"""Demo: AdaBoost with decision stumps on synthetic 2D data.

Generates points from a standard 2D Gaussian with the label given by
a diagonal boundary, ``x0 + x1 > 0.5``. No single axis-aligned stump
can separate this well (about 70% accuracy), but boosting combines
stumps on both axes to approximate the diagonal. Prints test accuracy,
the number of stumps used, the staged accuracy after 1, 5 and all
stumps, and the feature importances.
"""

import numpy as np

from adaboost_stumps.model import AdaBoostStumps


def make_data(seed: int = 42, n: int = 200):
    """Return (x, y): Gaussian points, label 1 where x0 + x1 > 0.5."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = (x[:, 0] + x[:, 1] > 0.5).astype(int)
    return x, y


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    single = AdaBoostStumps(n_estimators=1).fit(x_train, y_train)
    model = AdaBoostStumps(n_estimators=30).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    stages = model.staged_predict(x_test)
    stage_acc = [float(np.mean(s == y_test)) for s in stages]
    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Single stump test accuracy: {single.score(x_test, y_test):.3f}")
    print(f"Boosted test accuracy: {accuracy:.3f}")
    print(f"Stumps used: {model.n_estimators_} of {model.n_estimators}")
    print(f"Staged accuracy after 1 stump: {stage_acc[0]:.3f}")
    print(f"Staged accuracy after 5 stumps: {stage_acc[min(4, len(stage_acc) - 1)]:.3f}")
    print(f"Staged accuracy after all stumps: {stage_acc[-1]:.3f}")
    print(f"Feature importances: {np.round(model.feature_importances_, 3).tolist()}")


if __name__ == "__main__":
    main()
