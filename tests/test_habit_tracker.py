"""Tests for the habit_tracker Flask app, using Flask's test client."""
import pytest

from habit_tracker.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _create(client, name="Read", description=None):
    body = {"name": name}
    if description is not None:
        body["description"] = description
    return client.post("/habits", json=body)


def test_list_starts_empty(client):
    assert client.get("/habits").get_json() == []


def test_create_habit_returns_201_with_zero_streak(client):
    resp = _create(client, "Read", "Before bed")
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "name": "Read",
        "description": "Before bed",
        "done_dates": [],
        "total": 0,
        "streak": 0,
    }


def test_create_requires_non_empty_name(client):
    assert _create(client, "").status_code == 400
    assert _create(client, "   ").status_code == 400
    assert client.post("/habits", json={}).status_code == 400
    resp = client.post("/habits", json={"name": 42})
    assert resp.status_code == 400
    assert "name" in resp.get_json()["error"]


def test_create_rejects_non_string_description(client):
    resp = client.post("/habits", json={"name": "Run", "description": 5})
    assert resp.status_code == 400


def test_get_habit_and_404(client):
    habit_id = _create(client).get_json()["id"]
    assert client.get(f"/habits/{habit_id}").get_json()["name"] == "Read"
    assert client.get("/habits/999").status_code == 404


def test_mark_done_builds_streak(client):
    habit_id = _create(client).get_json()["id"]
    for day, expected in (("2026-11-11", 1), ("2026-11-12", 2), ("2026-11-13", 3)):
        resp = client.post(f"/habits/{habit_id}/done", json={"date": day})
        assert resp.status_code == 200
        assert resp.get_json()["streak"] == expected
    habit = client.get(f"/habits/{habit_id}").get_json()
    assert habit["total"] == 3
    assert habit["done_dates"] == ["2026-11-11", "2026-11-12", "2026-11-13"]


def test_gap_day_breaks_streak(client):
    habit_id = _create(client).get_json()["id"]
    client.post(f"/habits/{habit_id}/done", json={"date": "2026-11-10"})
    client.post(f"/habits/{habit_id}/done", json={"date": "2026-11-12"})
    client.post(f"/habits/{habit_id}/done", json={"date": "2026-11-13"})
    # Only the 12th-13th run counts; the 10th is separated by a gap.
    assert client.get(f"/habits/{habit_id}").get_json()["streak"] == 2


def test_mark_done_rejects_bad_and_duplicate_dates(client):
    habit_id = _create(client).get_json()["id"]
    assert client.post(
        f"/habits/{habit_id}/done", json={"date": "not-a-date"}
    ).status_code == 400
    assert client.post(
        f"/habits/{habit_id}/done", json={"date": "2026-02-30"}
    ).status_code == 400
    assert client.post(
        f"/habits/{habit_id}/done", json={"date": "2999-01-01"}
    ).status_code == 400
    client.post(f"/habits/{habit_id}/done", json={"date": "2026-11-13"})
    dup = client.post(f"/habits/{habit_id}/done", json={"date": "2026-11-13"})
    assert dup.status_code == 400
    assert "already" in dup.get_json()["error"]


def test_mark_done_on_unknown_habit_is_404(client):
    assert client.post("/habits/999/done", json={"date": "2026-11-13"}).status_code == 404


def test_unmark_done_and_errors(client):
    habit_id = _create(client).get_json()["id"]
    client.post(f"/habits/{habit_id}/done", json={"date": "2026-11-12"})
    client.post(f"/habits/{habit_id}/done", json={"date": "2026-11-13"})
    resp = client.delete(f"/habits/{habit_id}/done", json={"date": "2026-11-13"})
    assert resp.get_json()["streak"] == 1
    missing = client.delete(f"/habits/{habit_id}/done", json={"date": "2026-11-13"})
    assert missing.status_code == 400
    assert client.delete(f"/habits/{habit_id}/done", json={}).status_code == 400


def test_delete_habit(client):
    habit_id = _create(client).get_json()["id"]
    assert client.delete(f"/habits/{habit_id}").get_json() == {"deleted": habit_id}
    assert client.get(f"/habits/{habit_id}").status_code == 404
    assert client.delete("/habits/999").status_code == 404


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
