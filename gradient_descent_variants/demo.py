"""Demo: SGD vs momentum vs RMSprop vs Adam.

Two comparisons, both from the same starting point per problem:

1. Ill-conditioned quadratic, A = diag([1, 100]) — condition number 100.
   Plain SGD's single learning rate is throttled by the steep direction
   (lr must stay below 2/100), so it crawls along the flat direction;
   momentum accelerates through it and the adaptive methods rescale each
   coordinate, so all three beat SGD. Adam runs with beta2=0.99 here
   rather than the usual 0.999: with 0.999 the second moment remembers
   the huge initial gradients for ~1000 steps and Adam crawls; 0.99
   adapts in ~100 steps.
2. Binary logistic regression on synthetic data — all four converge to
   nearly the same loss, showing the adaptive methods are not magic, just
   better conditioned for badly scaled problems.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gradient_descent_variants.model import (
    SGD,
    Momentum,
    RMSprop,
    Adam,
    compare_optimizers,
    quadratic_loss,
    logistic_loss,
)


def _print_table(title, results, optimum=None):
    print(f"\n{title}")
    header = f"{'optimizer':<10} {'final loss':>12} {'loss drop':>12}"
    if optimum is not None:
        header += f" {'dist to optimum':>16}"
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        drop = res["history"][0] - res["loss"]
        row = f"{name:<10} {res['loss']:>12.6f} {drop:>12.6f}"
        if optimum is not None:
            row += f" {np.linalg.norm(res['x'] - optimum):>16.6f}"
        print(row)


def quadratic_demo():
    """Ill-conditioned quadratic: A = diag([1, 100]), optimum x* = A^-1 b."""
    A = np.diag([1.0, 100.0])
    b = np.array([1.0, 1.0])
    x_star = np.linalg.solve(A, b)
    objective = quadratic_loss(A, b)
    x0 = np.array([5.0, 5.0])
    print(f"Quadratic f(x) = 0.5 x^T A x - b^T x, A = diag([1, 100])")
    print(f"start x0 = {x0}, optimum x* = {x_star}, "
          f"optimum loss = {objective(x_star)[0]:.6f}")
    optimizers = {
        "SGD": SGD(lr=0.01),
        "Momentum": Momentum(lr=0.01, momentum=0.9),
        "RMSprop": RMSprop(lr=0.05),
        # beta2=0.99, not the 0.999 default: see module docstring.
        "Adam": Adam(lr=0.05, beta2=0.99),
    }
    results = compare_optimizers(objective, x0, n_steps=200, optimizers=optimizers)
    _print_table("200 steps from x0 = [5, 5]:", results, optimum=x_star)
    return results


def logistic_demo():
    """Logistic regression on synthetic, mildly overlapping classes."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    w_true = np.array([2.0, -1.0])
    p_true = 1.0 / (1.0 + np.exp(-(X @ w_true + 0.5)))
    y = (rng.random(n) < p_true).astype(float)
    # Append a ones column so the model learns an intercept too.
    X_aug = np.column_stack([X, np.ones(n)])
    objective = logistic_loss(X_aug, y)
    x0 = np.zeros(3)
    optimizers = {
        "SGD": SGD(lr=0.5),
        "Momentum": Momentum(lr=0.5, momentum=0.9),
        "RMSprop": RMSprop(lr=0.05),
        "Adam": Adam(lr=0.05),
    }
    results = compare_optimizers(objective, x0, n_steps=300, optimizers=optimizers)
    _print_table("Logistic regression, 300 steps from w0 = 0:", results)
    for name, res in results.items():
        acc = float(np.mean((X_aug @ res["x"] > 0) == (y == 1)))
        print(f"  {name:<10} train accuracy: {acc:.3f}")
    return results


def main():
    print("=" * 60)
    print("Gradient descent variants: SGD / Momentum / RMSprop / Adam")
    print("=" * 60)
    quad = quadratic_demo()
    logistic_demo()
    gains = {n: quad["SGD"]["loss"] - quad[n]["loss"]
             for n in ("Momentum", "RMSprop", "Adam")}
    print("\nTakeaway: on the ill-conditioned quadratic, after the same")
    print("200 steps, Momentum / RMSprop / Adam end "
          f"{gains['Momentum']:.2f} / {gains['RMSprop']:.2f} / {gains['Adam']:.2f}")
    print(f"lower in loss than plain SGD (SGD: {quad['SGD']['loss']:.3f}, "
          "optimum: -0.505); on well-scaled")
    print("logistic regression all four land within a whisker of each other.")


if __name__ == "__main__":
    main()
