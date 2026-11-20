"""Demo: matrix factorisation recommender on synthetic ratings.

Builds a low-rank ground truth (20 users x 15 items, rank 3, ratings
1-5), hides 15% of the entries as a test set, fits SGD matrix
factorisation on the rest, and prints train/test RMSE plus the top-5
recommendations for user 0.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from matrix_factorization.model import MatrixFactorization

SEED = 7
N_USERS, N_ITEMS, RANK = 20, 15, 3


def main():
    rng = np.random.default_rng(SEED)
    truth = rng.normal(0, 1, (N_USERS, RANK)) @ rng.normal(0, 1, (RANK, N_ITEMS))
    truth = np.clip(3.0 + truth, 1.0, 5.0)
    observed = rng.random((N_USERS, N_ITEMS)) < 0.85
    train = np.where(observed, truth, 0.0)
    test = np.where(~observed, truth, 0.0)

    model = MatrixFactorization(n_factors=3, learning_rate=0.02,
                                reg=0.02, n_epochs=150,
                                random_state=SEED).fit(train)
    print(f"users: {N_USERS}, items: {N_ITEMS}, "
          f"observed train ratings: {int(observed.sum())}, "
          f"held-out test ratings: {int((~observed).sum())}")
    print(f"train RMSE: {model.rmse(train):.4f}")
    print(f"test  RMSE: {model.rmse(test):.4f}")
    recs = model.recommend(0, n=5)
    print(f"top-5 recommendations for user 0: {recs}")
    print("predicted scores:",
          [round(model.predict(0, i), 2) for i in recs])
    return model


if __name__ == "__main__":
    main()
