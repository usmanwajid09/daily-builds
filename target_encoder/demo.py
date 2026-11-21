"""Demo: smoothed target encoding of a city feature vs conversion (0/1).

Cities have very different sample sizes and conversion rates. A rare city
with a perfect raw rate should NOT get an encoding of 1.0 -- smoothing
pulls it back toward the global mean.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from target_encoder.model import TargetEncoder


def main():
    rng = np.random.default_rng(seed=42)

    # (city, n_rows, true conversion rate)
    spec = [
        ("Lahore", 100, 0.60),
        ("Karachi", 60, 0.35),
        ("Islamabad", 30, 0.80),
        ("Quetta", 8, 1.00),   # rare, perfect raw rate
        ("Murree", 2, 0.00),   # very rare, zero raw rate
    ]
    cities, y = [], []
    for city, n, rate in spec:
        cities.extend([city] * n)
        y.extend(rng.random(n) < rate)
    X = np.array(cities)
    y = np.array(y, dtype=float)

    enc = TargetEncoder(smoothing=10.0).fit(X, y)

    print(f"Rows: {len(y)}   Global mean (conversion rate): "
          f"{enc.global_mean_:.4f}\n")
    print(f"{'City':<12}{'count':>7}{'raw mean':>10}{'smoothed':>11}")
    print("-" * 40)
    for city in enc.categories_[0]:
        mask = X == city
        raw_mean = y[mask].mean()
        print(f"{city:<12}{mask.sum():>7}{raw_mean:>10.4f}"
              f"{enc.encodings_[0][city]:>11.4f}")

    print("\nShrinkage in action:")
    print(f"  Quetta   (n=8,  raw mean 1.00) -> "
          f"{enc.encodings_[0]['Quetta']:.4f}  (pulled down toward global mean)")
    print(f"  Murree   (n=2,  raw mean 0.00) -> "
          f"{enc.encodings_[0]['Murree']:.4f}  (pulled up toward global mean)")
    print(f"  Lahore   (n=100, raw mean {y[X == 'Lahore'].mean():.4f}) -> "
          f"{enc.encodings_[0]['Lahore']:.4f}  (barely moved: plenty of data)")

    unseen = enc.transform(np.array(["Peshawar", "Lahore"]))
    print(f"\nUnseen category 'Peshawar' encodes to the global mean: "
          f"{unseen[0]:.4f} (Lahore -> {unseen[1]:.4f})")


if __name__ == "__main__":
    main()
