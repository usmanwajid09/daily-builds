import numpy as np
import pytest

from matrix_factorization import MatrixFactorization


def _low_rank_ratings(seed=0, n_users=12, n_items=10, rank=2, missing=0.4):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1, (n_users, rank)) @ rng.normal(0, 1, (rank, n_items))
    truth = np.clip(3.0 + truth, 1.0, 5.0)
    observed = rng.random((n_users, n_items)) >= missing
    return np.where(observed, truth, 0.0), truth


def _fit(seed=0, **kwargs):
    ratings, _ = _low_rank_ratings(seed=seed)
    params = dict(n_factors=2, n_epochs=120, random_state=seed)
    params.update(kwargs)
    return MatrixFactorization(**params).fit(ratings), ratings


def test_fit_reduces_rmse_below_global_mean_baseline():
    ratings, _ = _low_rank_ratings()
    model, _ = _fit()
    observed = ratings != 0
    baseline = float(np.sqrt(np.mean(
        (ratings[observed] - ratings[observed].mean()) ** 2)))
    assert model.rmse(ratings) < baseline
    assert model.rmse(ratings) < 0.6


def test_predict_all_shape_and_finite():
    model, ratings = _fit()
    pred = model.predict_all()
    assert pred.shape == ratings.shape
    assert np.isfinite(pred).all()


def test_predict_matches_predict_all():
    model, _ = _fit()
    assert model.predict(3, 4) == pytest.approx(model.predict_all()[3, 4])


def test_rmse_near_zero_on_rank_one_data():
    rng = np.random.default_rng(1)
    u = rng.normal(0, 1, (8, 1))
    v = rng.normal(0, 1, (1, 6))
    ratings = np.clip(3.0 + u @ v, 1.0, 5.0)
    model = MatrixFactorization(n_factors=1, learning_rate=0.05,
                                reg=0.0, n_epochs=400,
                                random_state=1).fit(ratings)
    assert model.rmse(ratings) < 0.05


def test_recommend_returns_n_unrated_items_sorted():
    model, ratings = _fit()
    recs = model.recommend(0, n=3)
    assert len(recs) == 3
    assert len(set(recs)) == 3
    assert all(ratings[0, i] == 0 for i in recs)
    scores = [model.predict(0, i) for i in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommend_include_rated_can_return_rated_items():
    model, _ = _fit()
    recs = model.recommend(0, n=5, exclude_rated=False)
    assert len(recs) == 5


def test_recommend_all_rated_user_raises():
    ratings = np.ones((3, 4)) * 4.0
    model = MatrixFactorization(n_epochs=5, random_state=0).fit(ratings)
    with pytest.raises(ValueError):
        model.recommend(0, n=2)


def test_recommend_n_too_large_raises():
    model, _ = _fit()
    with pytest.raises(ValueError):
        model.recommend(0, n=999)


def test_recommend_n_zero_raises():
    model, _ = _fit()
    with pytest.raises(ValueError):
        model.recommend(0, n=0)


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        MatrixFactorization().predict(0, 0)


def test_predict_out_of_range_user_raises():
    model, _ = _fit()
    with pytest.raises(ValueError):
        model.predict(999, 0)


def test_predict_out_of_range_item_raises():
    model, _ = _fit()
    with pytest.raises(ValueError):
        model.predict(0, 999)


def test_fit_empty_ratings_raises():
    with pytest.raises(ValueError):
        MatrixFactorization().fit(np.zeros((4, 5)))


def test_fit_1d_ratings_raises():
    with pytest.raises(ValueError):
        MatrixFactorization().fit(np.array([1, 2, 3]))


def test_fit_nan_entries_treated_as_missing():
    ratings, _ = _low_rank_ratings()
    ratings_nan = ratings.astype(float)
    ratings_nan[ratings_nan == 0] = np.nan
    model = MatrixFactorization(n_factors=2, n_epochs=60,
                                random_state=0).fit(ratings_nan)
    assert np.isfinite(model.predict_all()).all()


def test_fit_infinite_rating_raises():
    ratings = np.array([[5.0, np.inf], [0.0, 3.0]])
    with pytest.raises(ValueError):
        MatrixFactorization().fit(ratings)


def test_same_seed_reproducible():
    ratings, _ = _low_rank_ratings()
    m1 = MatrixFactorization(random_state=42, n_epochs=20).fit(ratings)
    m2 = MatrixFactorization(random_state=42, n_epochs=20).fit(ratings)
    assert np.allclose(m1.predict_all(), m2.predict_all())


def test_invalid_hyperparameters_raise():
    with pytest.raises(ValueError):
        MatrixFactorization(n_factors=0)
    with pytest.raises(ValueError):
        MatrixFactorization(learning_rate=0)
    with pytest.raises(ValueError):
        MatrixFactorization(reg=-1)
    with pytest.raises(ValueError):
        MatrixFactorization(n_epochs=0)
