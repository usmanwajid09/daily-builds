"""Tests for the time_tracker_api Flask app, using Flask's test client."""
import pytest

from time_tracker_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _project(client, name="NovaStack", rate=15, **extra):
    payload = {"name": name, "hourly_rate": rate}
    payload.update(extra)
    return client.post("/projects", json=payload)


def _entry(client, project_id, start="2026-11-21T09:00",
           end="2026-11-21T11:30", **extra):
    payload = {"start_time": start, "end_time": end}
    payload.update(extra)
    return client.post(f"/projects/{project_id}/entries", json=payload)


def test_list_starts_empty(client):
    assert client.get("/projects").get_json() == []


def test_create_project_returns_201(client):
    resp = _project(client, description="Client work")
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "name": "NovaStack",
        "hourly_rate": 15.0,
        "description": "Client work",
    }


def test_hourly_rate_defaults_to_zero(client):
    assert client.post("/projects", json={"name": "Study"}).get_json()[
        "hourly_rate"] == 0.0


def test_create_missing_name_400(client):
    assert client.post("/projects", json={"hourly_rate": 10}).status_code == 400


def test_create_negative_rate_400(client):
    assert _project(client, rate=-5).status_code == 400
    assert _project(client, rate="free").status_code == 400


def test_duplicate_name_409(client):
    assert _project(client).status_code == 201
    assert _project(client, name="novastack").status_code == 409


def test_get_project(client):
    _project(client)
    assert client.get("/projects/1").get_json()["name"] == "NovaStack"


def test_get_missing_project_404(client):
    assert client.get("/projects/99").status_code == 404


def test_update_project(client):
    _project(client)
    resp = client.put("/projects/1", json={"hourly_rate": 20})
    assert resp.status_code == 200
    assert resp.get_json()["hourly_rate"] == 20.0


def test_update_to_duplicate_name_409(client):
    _project(client, name="A")
    _project(client, name="B")
    assert client.put("/projects/2", json={"name": "a"}).status_code == 409


def test_update_nothing_400(client):
    _project(client)
    assert client.put("/projects/1", json={}).status_code == 400


def test_delete_project_cascades_entries(client):
    _project(client)
    _entry(client, 1)
    assert client.delete("/projects/1").get_json() == {"deleted": 1}
    assert client.get("/entries/1").status_code == 404


def test_create_entry_returns_201_with_duration(client):
    _project(client)
    resp = _entry(client, 1, description="Landing page")
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["duration_minutes"] == 150.0
    assert body["running"] is False
    assert body["start_time"] == "2026-11-21T09:00"
    assert body["end_time"] == "2026-11-21T11:30"


def test_create_entry_end_before_start_400(client):
    _project(client)
    assert _entry(client, 1, start="2026-11-21T11:00",
                  end="2026-11-21T10:00").status_code == 400


def test_create_entry_bad_time_400(client):
    _project(client)
    assert _entry(client, 1, start="yesterday",
                  end="2026-11-21T10:00").status_code == 400
    assert _entry(client, 1, start="2026-11-21T09:00",
                  end=None).status_code == 400


def test_create_entry_missing_project_404(client):
    assert _entry(client, 99).status_code == 404


def test_start_and_stop_timer(client):
    _project(client)
    started = client.post("/projects/1/entries/start",
                          json={"description": "Work",
                                "start_time": "2026-11-21T09:00"})
    assert started.status_code == 201
    assert started.get_json()["running"] is True
    assert started.get_json()["duration_minutes"] is None
    stopped = client.post("/entries/1/stop",
                          json={"end_time": "2026-11-21T10:30"})
    assert stopped.status_code == 200
    assert stopped.get_json()["duration_minutes"] == 90.0
    assert stopped.get_json()["running"] is False


def test_start_second_timer_409(client):
    _project(client)
    client.post("/projects/1/entries/start", json={})
    resp = client.post("/projects/1/entries/start", json={})
    assert resp.status_code == 409


def test_stop_already_stopped_400(client):
    _project(client)
    _entry(client, 1)
    assert client.post("/entries/1/stop", json={}).status_code == 400


def test_stop_end_before_start_400(client):
    _project(client)
    client.post("/projects/1/entries/start",
                json={"start_time": "2026-11-21T10:00"})
    resp = client.post("/entries/1/stop",
                       json={"end_time": "2026-11-21T09:00"})
    assert resp.status_code == 400


def test_stop_missing_entry_404(client):
    assert client.post("/entries/99/stop", json={}).status_code == 404


def test_list_entries_sorted_by_start(client):
    _project(client)
    _entry(client, 1, start="2026-11-21T14:00", end="2026-11-21T15:00")
    _entry(client, 1, start="2026-11-21T09:00", end="2026-11-21T10:00")
    rows = client.get("/projects/1/entries").get_json()
    assert [r["start_time"] for r in rows] == ["2026-11-21T09:00",
                                               "2026-11-21T14:00"]


def test_delete_entry(client):
    _project(client)
    _entry(client, 1)
    assert client.delete("/entries/1").get_json() == {"deleted": 1}
    assert client.get("/entries/1").status_code == 404


def test_project_summary_totals_and_earnings(client):
    _project(client, rate=20)
    _entry(client, 1, start="2026-11-21T09:00", end="2026-11-21T11:00")
    _entry(client, 1, start="2026-11-21T14:00", end="2026-11-21T14:30")
    summary = client.get("/projects/1/summary").get_json()
    assert summary["total_minutes"] == 150.0
    assert summary["total_hours"] == 2.5
    assert summary["entry_count"] == 2
    assert summary["completed_entries"] == 2
    assert summary["running"] is False
    assert summary["earnings"] == 50.0


def test_project_summary_running_flag(client):
    _project(client)
    client.post("/projects/1/entries/start", json={})
    summary = client.get("/projects/1/summary").get_json()
    assert summary["running"] is True
    assert summary["total_minutes"] == 0


def test_overall_summary(client):
    _project(client, name="A", rate=10)
    _project(client, name="B", rate=0)
    _entry(client, 1, start="2026-11-21T09:00", end="2026-11-21T10:00")
    _entry(client, 2, start="2026-11-21T09:00", end="2026-11-21T11:00")
    overall = client.get("/summary").get_json()
    assert overall["project_count"] == 2
    assert overall["total_minutes"] == 180.0
    assert overall["total_hours"] == 3.0
    assert overall["total_earnings"] == 10.0
    assert overall["running_projects"] == 0


def test_unknown_route_404(client):
    assert client.get("/nope").status_code == 404
