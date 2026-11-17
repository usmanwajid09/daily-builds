"""Demo: fit a CART regression tree to a noisy sine wave and report fit quality."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_tree_regressor.model import DecisionTreeRegressor
import numpy as np

rng = np.random.default_rng(42)
n = 150
X = rng.uniform(0, 2 * np.pi, size=(n, 1))
y = np.sin(X[:, 0]) + rng.normal(0, 0.1, size=n)

X_train, X_test = X[:120], X[120:]
y_train, y_test = y[:120], y[120:]

tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
r2 = tree.score(X_test, y_test)
mse = float(np.mean((y_test - y_pred) ** 2))

print(f"Test R²: {r2:.4f}")
print(f"Test MSE: {mse:.4f}")
print(f"Tree depth: {tree.max_depth_}")
print(f"Leaf count: {tree.n_leaves_}")

assert r2 > 0.85, f"Test R² {r2:.4f} did not exceed 0.85"
print("Demo passed: test R² > 0.85")
