"""Tests for the review_api Flask app, using Flask's test client."""
import pytest

from review_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _review(client, product="widget-1", author="Alice", rating=5, **extra):
    payload = {"product_id": product, "author": author, "rating": rating}
    payload.update(extra)
    return client.post("/reviews", json=payload)


def test_create_review_returns_201(client):
    resp = _review(client, title="Great", comment="Loved it.")
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["id"] == 1
    assert data["helpful_votes"] == 0
    assert data["flagged"] is False
    assert data["status"] == "visible"


def test_create_review_missing_author_returns_400(client):
    resp = client.post("/reviews", json={"product_id": "widget-1", "rating": 5})
    assert resp.status_code == 400
    assert "author" in resp.get_json()["error"]


def test_create_review_rating_below_1_returns_400(client):
    assert _review(client, rating=0).status_code == 400


def test_create_review_rating_above_5_returns_400(client):
    assert _review(client, rating=6).status_code == 400


def test_create_review_non_integer_rating_returns_400(client):
    assert _review(client, rating=4.5).status_code == 400
    assert _review(client, rating="5").status_code == 400


def test_create_review_duplicate_author_and_product_returns_409(client):
    _review(client)
    resp = _review(client, rating=3)
    assert resp.status_code == 409
    assert "already reviewed" in resp.get_json()["error"]


def test_same_author_can_review_a_different_product(client):
    assert _review(client).status_code == 201
    assert _review(client, product="widget-2").status_code == 201


def test_list_reviews_filters_by_product_and_status(client):
    _review(client, rating=5)
    _review(client, author="Bob", rating=4)
    _review(client, product="widget-2", author="Carol", rating=3)
    assert len(client.get("/reviews").get_json()) == 3
    by_product = client.get("/reviews", query_string={"product_id": "widget-2"}).get_json()
    assert len(by_product) == 1
    visible = client.get("/reviews", query_string={"status": "visible"}).get_json()
    assert len(visible) == 3


def test_list_reviews_invalid_status_returns_400(client):
    assert client.get("/reviews", query_string={"status": "nope"}).status_code == 400


def test_get_review_and_missing_review_404(client):
    review = _review(client).get_json()
    assert client.get(f"/reviews/{review['id']}").get_json()["author"] == "Alice"
    assert client.get("/reviews/99").status_code == 404


def test_summary_average_and_distribution(client):
    _review(client, rating=5)
    _review(client, author="Bob", rating=4)
    _review(client, author="Carol", rating=5)
    summary = client.get("/products/widget-1/summary").get_json()
    assert summary["count"] == 3
    assert summary["average_rating"] == 4.67
    assert summary["distribution"] == {"1": 0, "2": 0, "3": 0, "4": 1, "5": 2}


def test_summary_for_product_with_no_reviews_returns_404(client):
    assert client.get("/products/nothing/summary").status_code == 404


def test_helpful_vote_increments(client):
    review = _review(client).get_json()
    client.post(f"/reviews/{review['id']}/helpful")
    resp = client.post(f"/reviews/{review['id']}/helpful")
    assert resp.get_json()["helpful_votes"] == 2


def test_helpful_vote_missing_review_returns_404(client):
    assert client.post("/reviews/99/helpful").status_code == 404


def test_flag_then_approve_restores_visible(client):
    review = _review(client).get_json()
    flagged = client.post(f"/reviews/{review['id']}/flag").get_json()
    assert flagged["flagged"] is True
    assert flagged["status"] == "flagged"
    approved = client.post(
        f"/reviews/{review['id']}/moderate", json={"action": "approve"}
    ).get_json()
    assert approved["flagged"] is False
    assert approved["status"] == "visible"


def test_flag_then_hide_excludes_from_summary(client):
    _review(client, rating=5)
    spam = _review(client, author="Spammer", rating=1).get_json()
    client.post(f"/reviews/{spam['id']}/flag")
    hidden = client.post(
        f"/reviews/{spam['id']}/moderate", json={"action": "hide"}
    ).get_json()
    assert hidden["status"] == "hidden"
    summary = client.get("/products/widget-1/summary").get_json()
    assert summary["count"] == 1
    assert summary["average_rating"] == 5.0


def test_moderate_unflagged_review_returns_400(client):
    review = _review(client).get_json()
    resp = client.post(f"/reviews/{review['id']}/moderate", json={"action": "hide"})
    assert resp.status_code == 400
    assert "not flagged" in resp.get_json()["error"]


def test_moderate_invalid_action_returns_400(client):
    review = _review(client).get_json()
    client.post(f"/reviews/{review['id']}/flag")
    resp = client.post(
        f"/reviews/{review['id']}/moderate", json={"action": "delete"}
    )
    assert resp.status_code == 400


def test_flag_hidden_review_returns_400(client):
    review = _review(client).get_json()
    client.post(f"/reviews/{review['id']}/flag")
    client.post(f"/reviews/{review['id']}/moderate", json={"action": "hide"})
    assert client.post(f"/reviews/{review['id']}/flag").status_code == 400


def test_delete_review(client):
    review = _review(client).get_json()
    assert client.delete(f"/reviews/{review['id']}").status_code == 200
    assert client.get(f"/reviews/{review['id']}").status_code == 404
    assert client.get("/products/widget-1/summary").status_code == 404
