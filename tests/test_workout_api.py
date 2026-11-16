"""Tests for the workout_api Flask app, using Flask's test client."""
import pytest

from workout_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _log(client, workout_type="run", duration=30, calories=300, date=None):
    payload = {
        "type": workout_type,
        "duration_minutes": duration,
        "calories": calories,
    }
    if date is not None:
        payload["date"] = date
    return client.post("/workouts", json=payload)


def test_list_starts_empty(client):
    assert client.get("/workouts").get_json() == []


def test_create_workout_returns_201_with_id(client):
    resp = _log(client, date="2026-11-16")
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["id"] == 1
    assert body["type"] == "run"
    assert body["date"] == "2026-11-16"


def test_create_defaults_date_to_today(client):
    from datetime import date

    body = _log(client).get_json()
    assert body["date"] == date.today().isoformat()


def test_create_trims_type(client):
    body = _log(client, workout_type="  Swim ").get_json()
    assert body["type"] == "Swim"


def test_create_rejects_bad_fields(client):
    assert _log(client, workout_type=" ").status_code == 400
    assert _log(client, duration=0).status_code == 400
    assert _log(client, duration=-10).status_code == 400
    assert _log(client, duration="30").status_code == 400
    assert _log(client, calories=-1).status_code == 400
    assert _log(client, date="16-11-2026").status_code == 400
    assert _log(client, date="2026-13-01").status_code == 400


def test_create_rejects_non_json_body(client):
    resp = client.post("/workouts", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_filter_by_type(client):
    _log(client, "run", date="2026-11-16")
    _log(client, "swim", date="2026-11-16")
    _log(client, "run", date="2026-11-17")
    runs = client.get("/workouts?type=RUN").get_json()
    assert len(runs) == 2
    assert all(w["type"] == "run" for w in runs)
    assert client.get("/workouts?type=yoga").get_json() == []


def test_weekly_totals_cover_monday_to_sunday(client):
    _log(client, "run", 30, 320, "2026-11-16")  # Monday
    _log(client, "swim", 45, 400, "2026-11-18")  # Wednesday
    _log(client, "run", 20, 200, "2026-11-23")  # next Monday, outside the week
    weekly = client.get("/workouts/weekly?week=2026-11-18").get_json()
    assert weekly["week_start"] == "2026-11-16"
    assert weekly["week_end"] == "2026-11-22"
    assert weekly["count"] == 2
    assert weekly["total_duration_minutes"] == 75
    assert weekly["total_calories"] == 720


def test_weekly_empty_week_is_zero(client):
    weekly = client.get("/workouts/weekly?week=2026-11-16").get_json()
    assert weekly == {
        "week_start": "2026-11-16",
        "week_end": "2026-11-22",
        "count": 0,
        "total_duration_minutes": 0,
        "total_calories": 0,
    }


def test_weekly_rejects_bad_week_param(client):
    assert client.get("/workouts/weekly?week=not-a-date").status_code == 400


def test_summary_by_type(client):
    _log(client, "run", 30, 320, "2026-11-16")
    _log(client, "run", 25, 260, "2026-11-17")
    _log(client, "swim", 45, 400, "2026-11-18")
    summary = client.get("/workouts/summary").get_json()
    assert summary == {
        "run": {"count": 2, "total_duration_minutes": 55, "total_calories": 580},
        "swim": {"count": 1, "total_duration_minutes": 45, "total_calories": 400},
    }


def test_get_and_delete_round_trip(client):
    workout_id = _log(client, date="2026-11-16").get_json()["id"]
    assert client.get(f"/workouts/{workout_id}").get_json()["type"] == "run"
    assert client.delete(f"/workouts/{workout_id}").get_json() == {"deleted": workout_id}
    assert client.get(f"/workouts/{workout_id}").status_code == 404


def test_unknown_workout_is_404_json(client):
    for method in ("get", "delete"):
        resp = getattr(client, method)("/workouts/999", json={})
        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"]


def test_unknown_route_is_404_json(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert "error" in resp.get_json()
