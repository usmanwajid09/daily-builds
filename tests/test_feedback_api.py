"""Tests for the feedback_api Flask app, using Flask's test client."""
import pytest

from feedback_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _submit(client, rating, comment="Great!", name=None):
    payload = {"rating": rating, "comment": comment}
    if name is not None:
        payload["name"] = name
    return client.post("/feedback", json=payload)


def test_list_starts_empty(client):
    assert client.get("/feedback").get_json() == []


def test_summary_empty(client):
    summary = client.get("/feedback/summary").get_json()
    assert summary == {
        "count": 0,
        "average_rating": None,
        "distribution": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
    }


def test_create_feedback_returns_201_with_id(client):
    resp = _submit(client, 5, "Brilliant service.", name="Ayesha")
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["id"] == 1
    assert body["rating"] == 5
    assert body["name"] == "Ayesha"


def test_create_trims_comment_and_name(client):
    body = _submit(client, 4, "  Good  ", name="  Bilal ").get_json()
    assert body["comment"] == "Good"
    assert body["name"] == "Bilal"


def test_create_rejects_bad_ratings(client):
    for bad in (0, 6, -1, 3.5, "5", True, None):
        assert _submit(client, bad).status_code == 400


def test_create_rejects_missing_or_blank_comment(client):
    assert client.post("/feedback", json={"rating": 5}).status_code == 400
    assert _submit(client, 5, "   ").status_code == 400
    assert client.post(
        "/feedback", json={"rating": 5, "comment": "ok", "name": " "}
    ).status_code == 400


def test_create_rejects_non_json_body(client):
    resp = client.post("/feedback", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_summary_average_and_distribution(client):
    _submit(client, 5)
    _submit(client, 4)
    _submit(client, 5)
    _submit(client, 2)
    summary = client.get("/feedback/summary").get_json()
    assert summary["count"] == 4
    assert summary["average_rating"] == 4.0
    assert summary["distribution"] == {"1": 0, "2": 1, "3": 0, "4": 1, "5": 2}


def test_filter_by_rating(client):
    _submit(client, 5, "Loved it")
    _submit(client, 3, "Meh")
    _submit(client, 5, "Great")
    five = client.get("/feedback?rating=5").get_json()
    assert [e["rating"] for e in five] == [5, 5]
    assert len(client.get("/feedback?rating=3").get_json()) == 1
    assert client.get("/feedback?rating=1").get_json() == []


def test_filter_rejects_bad_rating_param(client):
    for bad in ("0", "6", "abc", "4.5"):
        assert client.get(f"/feedback?rating={bad}").status_code == 400


def test_get_and_delete_round_trip(client):
    entry_id = _submit(client, 4, "Good").get_json()["id"]
    assert client.get(f"/feedback/{entry_id}").get_json()["rating"] == 4
    assert client.delete(f"/feedback/{entry_id}").get_json() == {"deleted": entry_id}
    assert client.get(f"/feedback/{entry_id}").status_code == 404
    assert client.get("/feedback/summary").get_json()["count"] == 0


def test_unknown_entry_is_404_json(client):
    for method in ("get", "delete"):
        resp = getattr(client, method)("/feedback/999", json={})
        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"]


def test_unknown_route_is_404_json(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert "error" in resp.get_json()
