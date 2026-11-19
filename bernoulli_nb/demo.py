"""Builds a tiny binary word-presence dataset, fits Bernoulli naive Bayes, reports accuracy."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bernoulli_nb.model import BernoulliNB

# Vocabulary: ["free", "win", "money", "meeting", "project", "lunch"]
# Spam messages mention free/win/money; ham messages mention meeting/project/lunch.
VOCAB = ["free", "win", "money", "meeting", "project", "lunch"]


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 100
    x_spam = rng.random((n, 6)) < np.array([0.8, 0.7, 0.75, 0.05, 0.05, 0.1])
    x_ham = rng.random((n, 6)) < np.array([0.05, 0.05, 0.1, 0.8, 0.75, 0.7])
    x = np.vstack([x_spam, x_ham]).astype(int)
    y = np.array([1] * n + [0] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = BernoulliNB(alpha=1.0).fit(x_train, y_train)
    preds = model.predict(x_test)
    accuracy = float(np.mean(preds == y_test))

    print(f"Vocabulary: {VOCAB}")
    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"P(word=1 | spam): {np.round(np.exp(model.feature_log_prob_[1]), 3).tolist()}")
    print(f"P(word=1 | ham):  {np.round(np.exp(model.feature_log_prob_[0]), 3).tolist()}")
    print(f"First 5 predictions: {preds[:5].tolist()}")
    print(f"First 5 true labels: {y_test[:5].tolist()}")


if __name__ == "__main__":
    main()
