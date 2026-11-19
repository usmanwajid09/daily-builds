"""Demo: random-forest regression on synthetic data.

Fits a forest to y = 10*sin(x0) + 5*x1^2 + noise, where x2 is pure
noise, then reports the train R^2, the out-of-bag R^2, and the feature
importances (x0 and x1 should dominate x2).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from random_forest_regressor.model import RandomForestRegressor


def make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(n, 3))
    y = 10.0 * np.sin(x[:, 0]) + 5.0 * x[:, 1] ** 2 + rng.normal(0, 1.0, size=n)
    return x, y


def main():
    x, y = make_data()
    model = RandomForestRegressor(n_estimators=30, max_depth=8,
                                  max_features=None, random_state=42)
    model.fit(x, y)
    print(f"Samples: {x.shape[0]}, features: {x.shape[1]}, trees: {model.n_estimators}")
    print(f"Train R^2: {model.score(x, y):.4f}")
    print(f"OOB R^2:   {model.oob_score_:.4f}")
    print("Feature importances:")
    for i, imp in enumerate(model.feature_importances_):
        print(f"  x{i}: {imp:.3f}")
    x_new = np.array([[0.5, 1.0, 0.0], [-2.0, 0.5, 1.0]])
    print(f"Predictions for {x_new.tolist()}: "
          f"{np.round(model.predict(x_new), 3).tolist()}")


if __name__ == "__main__":
    main()
