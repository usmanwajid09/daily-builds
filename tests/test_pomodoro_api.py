"""Tests for the pomodoro_api Flask app, using Flask's test client."""
import pytest

from pomodoro_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _add(client, task="Write report", **extra):
    payload = {"task": task}
    payload.update(extra)
    return client.post("/sessions", json=payload)


def test_list_starts_empty(client):
    assert client.get("/sessions").get_json() == []


def test_create_defaults_focus_25_min(client):
    resp = _add(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["type"] == "focus"
    assert body["duration_minutes"] == 25
    assert body["status"] == "pending"
    assert body["started_at"] is None
    assert body["completed_at"] is None
    assert body["created_at"].endswith("Z")
    assert body["id"] == 1


def test_create_break_defaults(client):
    short = _add(client, task="Stretch", type="short_break").get_json()
    assert short["duration_minutes"] == 5
    long_break = _add(client, task="Lunch", type="long_break").get_json()
    assert long_break["duration_minutes"] == 15


def test_create_custom_duration(client):
    body = _add(client, type="focus", duration_minutes=50).get_json()
    assert body["duration_minutes"] == 50


def test_create_missing_task_400(client):
    resp = client.post("/sessions", json={"type": "focus"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_create_blank_task_400(client):
    assert _add(client, task="   ").status_code == 400


def test_create_bad_type_400(client):
    assert _add(client, type="nap").status_code == 400


def test_create_bad_duration_400(client):
    for bad in (0, -5, 241, 25.5, "25", True):
        assert _add(client, duration_minutes=bad).status_code == 400


def test_create_non_json_400(client):
    resp = client.post("/sessions", data="not json",
                       content_type="application/json")
    assert resp.status_code == 400


def test_get_session_404(client):
    assert client.get("/sessions/99").status_code == 404


def test_start_complete_flow(client):
    session = _add(client).get_json()
    started = client.post(f"/sessions/{session['id']}/start").get_json()
    assert started["status"] == "running"
    assert started["started_at"] is not None
    done = client.post(f"/sessions/{session['id']}/complete").get_json()
    assert done["status"] == "completed"
    assert done["completed_at"] is not None


def test_start_twice_409(client):
    session = _add(client).get_json()
    client.post(f"/sessions/{session['id']}/start")
    resp = client.post(f"/sessions/{session['id']}/start")
    assert resp.status_code == 409


def test_start_completed_409(client):
    session = _add(client).get_json()
    client.post(f"/sessions/{session['id']}/start")
    client.post(f"/sessions/{session['id']}/complete")
    assert client.post(f"/sessions/{session['id']}/start").status_code == 409


def test_complete_pending_409(client):
    session = _add(client).get_json()
    assert client.post(f"/sessions/{session['id']}/complete").status_code == 409


def test_complete_missing_404(client):
    assert client.post("/sessions/99/complete").status_code == 404


def test_cancel_pending_and_running(client):
    pending = _add(client, task="A").get_json()
    assert client.post(f"/sessions/{pending['id']}/cancel").get_json()["status"] \
        == "cancelled"
    running = _add(client, task="B").get_json()
    client.post(f"/sessions/{running['id']}/start")
    assert client.post(f"/sessions/{running['id']}/cancel").get_json()["status"] \
        == "cancelled"


def test_cancel_completed_409(client):
    session = _add(client).get_json()
    client.post(f"/sessions/{session['id']}/start")
    client.post(f"/sessions/{session['id']}/complete")
    assert client.post(f"/sessions/{session['id']}/cancel").status_code == 409


def test_delete_session(client):
    session = _add(client).get_json()
    assert client.delete(f"/sessions/{session['id']}").get_json() == {
        "deleted": session["id"]
    }
    assert client.get(f"/sessions/{session['id']}").status_code == 404
    assert client.delete("/sessions/99").status_code == 404


def test_list_filters(client):
    _add(client, task="A", type="focus")
    _add(client, task="B", type="short_break")
    running = _add(client, task="C", type="focus").get_json()
    client.post(f"/sessions/{running['id']}/start")
    assert len(client.get("/sessions").get_json()) == 3
    assert len(client.get("/sessions", query_string={"type": "focus"}).get_json()) == 2
    assert len(client.get("/sessions", query_string={"status": "running"}).get_json()) == 1
    assert client.get("/sessions", query_string={"status": "bogus"}).status_code == 400
    assert client.get("/sessions", query_string={"type": "bogus"}).status_code == 400


def test_list_date_filter(client):
    session = _add(client).get_json()
    today = session["created_at"][:10]
    assert len(client.get("/sessions", query_string={"date": today}).get_json()) == 1
    assert client.get("/sessions", query_string={"date": "1999-01-01"}).get_json() == []
    assert client.get("/sessions", query_string={"date": "not-a-date"}).status_code == 400


def test_stats_totals(client):
    _add(client, task="A", type="focus")
    _add(client, task="B", type="short_break")
    third = _add(client, task="C", type="focus").get_json()
    client.post(f"/sessions/{third['id']}/start")
    client.post(f"/sessions/{third['id']}/complete")
    stats = client.get("/stats").get_json()
    assert stats["total_sessions"] == 3
    assert stats["completed_sessions"] == 1
    assert stats["focus_minutes_completed"] == 25
    assert stats["by_type"] == {"focus": 2, "short_break": 1, "long_break": 0}
    assert stats["by_status"]["completed"] == 1
    assert stats["by_status"]["pending"] == 2
    assert stats["today"]["total_sessions"] == 3
    assert stats["today"]["focus_minutes_completed"] == 25


def test_stats_empty(client):
    stats = client.get("/stats").get_json()
    assert stats["total_sessions"] == 0
    assert stats["focus_minutes_completed"] == 0


def test_daily_stats(client):
    _add(client, task="A", type="focus")
    done = _add(client, task="B", type="focus", duration_minutes=50).get_json()
    client.post(f"/sessions/{done['id']}/start")
    client.post(f"/sessions/{done['id']}/complete")
    today = done["created_at"][:10]
    daily = client.get("/stats/daily", query_string={"date": today}).get_json()
    assert daily["date"] == today
    assert daily["total_sessions"] == 2
    assert daily["completed_sessions"] == 1
    assert daily["focus_minutes_completed"] == 50
    assert daily["by_type"]["focus"] == 2
    empty = client.get("/stats/daily", query_string={"date": "1999-01-01"}).get_json()
    assert empty["total_sessions"] == 0
    assert empty["focus_minutes_completed"] == 0


def test_daily_stats_defaults_to_today(client):
    _add(client, task="A")
    daily = client.get("/stats/daily").get_json()
    assert daily["total_sessions"] == 1
    assert client.get("/stats/daily", query_string={"date": "bogus"}).status_code == 400


def test_unknown_route_404(client):
    assert client.get("/nope").status_code == 404
