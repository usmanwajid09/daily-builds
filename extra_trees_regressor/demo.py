"""Demo: extra trees regressor on synthetic noisy sinusoidal data.

Fits a 100-tree forest and a single extra tree on y = sin(x) + noise,
prints test R^2 for both and the forest's feature importances.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from extra_trees_regressor.model import ExtraTreesRegressor


def main():
    rng = np.random.default_rng(42)
    x = rng.uniform(-3, 3, size=(300, 2))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] + rng.normal(0, 0.1, size=300)
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    forest = ExtraTreesRegressor(n_estimators=100, random_state=0)
    forest.fit(x_train, y_train)
    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Extra trees (100 trees) test R^2: {forest.score(x_test, y_test):.3f}")

    single = ExtraTreesRegressor(n_estimators=1, random_state=0)
    single.fit(x_train, y_train)
    print(f"Single extra tree test R^2: {single.score(x_test, y_test):.3f}")
    print("Feature importances:",
          np.round(forest.feature_importances_, 3).tolist())


if __name__ == "__main__":
    main()
