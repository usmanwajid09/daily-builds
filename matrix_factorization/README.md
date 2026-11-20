# Matrix Factorisation Recommender (SGD)

Learn low-rank user and item factors (plus global, user and item bias
terms) from a sparse ratings matrix by stochastic gradient descent.
Missing entries are encoded as `0` (or `NaN`) and ignored during
training. Given a fitted model you can predict any rating, score the
whole matrix, or ask for a user's top-N recommendations, excluding
items they already rated.

## Usage

```python
import numpy as np
from matrix_factorization import MatrixFactorization

ratings = np.array([[5, 0, 3],
                    [0, 4, 0],
                    [2, 0, 5]], dtype=float)  # 0 = not rated
model = MatrixFactorization(n_factors=3, n_epochs=100,
                            random_state=0).fit(ratings)
print(model.predict(0, 1))        # predicted rating for user 0, item 1
print(model.rmse(ratings))        # RMSE on the observed entries
print(model.recommend(0, n=2))    # top-2 unrated items for user 0
```

## Run demo / tests

```bash
python matrix_factorization/demo.py
python -m pytest tests/test_matrix_factorization.py -v
```

The demo builds a synthetic low-rank ratings matrix (20 users x 15
items, rank 3), holds out 15% of the ratings, fits the model, and
prints train/test RMSE plus the top-5 recommendations for user 0.
