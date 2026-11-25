"""Tests for the study planner API."""
import pytest

from study_planner_api.app import create_app


@pytest.fixture()
def client():
    return create_app().test_client()


def _subject(client, name="Maths", goal=60):
    resp = client.post("/subjects", json={"name": name, "daily_goal_minutes": goal})
    assert resp.status_code == 201
    return resp.get_json()


def _session(client, subject_id, minutes, date="2026-11-25", topic="Algebra"):
    resp = client.post("/sessions", json={
        "subject_id": subject_id, "minutes": minutes, "date": date, "topic": topic,
    })
    assert resp.status_code == 201
    return resp.get_json()


def test_create_subject(client):
    subject = _subject(client)
    assert subject["name"] == "Maths"
    assert subject["daily_goal_minutes"] == 60
    assert subject["id"] == 1


def test_create_subject_missing_name(client):
    resp = client.post("/subjects", json={"daily_goal_minutes": 60})
    assert resp.status_code == 400


def test_create_subject_bad_goal(client):
    assert client.post("/subjects", json={"name": "X", "daily_goal_minutes": 0}).status_code == 400
    assert client.post("/subjects", json={"name": "X", "daily_goal_minutes": -5}).status_code == 400
    assert client.post("/subjects", json={"name": "X", "daily_goal_minutes": "60"}).status_code == 400


def test_get_and_list_subjects(client):
    subject = _subject(client)
    assert client.get(f"/subjects/{subject['id']}").get_json()["name"] == "Maths"
    assert len(client.get("/subjects").get_json()) == 1


def test_get_missing_subject_404(client):
    assert client.get("/subjects/99").status_code == 404


def test_log_session_and_totals(client):
    subject = _subject(client)
    _session(client, subject["id"], 40)
    _session(client, subject["id"], 30, topic="Calculus")
    _session(client, subject["id"], 60, date="2026-11-24")
    totals = client.get(f"/subjects/{subject['id']}/totals").get_json()
    assert totals["total_minutes"] == 130
    assert totals["session_count"] == 3
    assert totals["days_studied"] == 2
    assert totals["average_minutes_per_day"] == 65.0


def test_session_unknown_subject_404(client):
    resp = client.post("/sessions", json={"subject_id": 99, "minutes": 30, "date": "2026-11-25"})
    assert resp.status_code == 404


def test_session_bad_minutes_400(client):
    subject = _subject(client)
    assert client.post("/sessions", json={"subject_id": subject["id"], "minutes": 0}).status_code == 400
    assert client.post("/sessions", json={"subject_id": subject["id"], "minutes": -5}).status_code == 400


def test_session_bad_date_400(client):
    subject = _subject(client)
    resp = client.post("/sessions", json={
        "subject_id": subject["id"], "minutes": 30, "date": "25-11-2026",
    })
    assert resp.status_code == 400


def test_list_sessions_filtered(client):
    subject = _subject(client)
    other = _subject(client, "Physics", 45)
    _session(client, subject["id"], 40)
    _session(client, other["id"], 50)
    mine = client.get(f"/sessions?subject_id={subject['id']}").get_json()
    assert len(mine) == 1 and mine[0]["subject_id"] == subject["id"]


def test_daily_goal_check(client):
    maths = _subject(client, "Maths", 60)
    physics = _subject(client, "Physics", 45)
    _session(client, maths["id"], 40)
    _session(client, maths["id"], 30)
    _session(client, physics["id"], 50)
    goals = client.get("/goals/daily?date=2026-11-25").get_json()
    assert goals["total_minutes"] == 120
    assert goals["goals_met"] == 2
    assert goals["all_met"] is True
    by_name = {row["name"]: row for row in goals["subjects"]}
    assert by_name["Maths"]["studied_minutes"] == 70
    assert by_name["Physics"]["studied_minutes"] == 50


def test_daily_goal_not_met(client):
    subject = _subject(client, "Maths", 120)
    _session(client, subject["id"], 30)
    goals = client.get("/goals/daily?date=2026-11-25").get_json()
    assert goals["all_met"] is False
    assert goals["subjects"][0]["met"] is False


def test_delete_subject_removes_sessions(client):
    subject = _subject(client)
    _session(client, subject["id"], 40)
    assert client.delete(f"/subjects/{subject['id']}").status_code == 200
    assert client.get("/sessions").get_json() == []
    assert client.get(f"/subjects/{subject['id']}").status_code == 404


def test_delete_session(client):
    subject = _subject(client)
    session = _session(client, subject["id"], 40)
    assert client.delete(f"/sessions/{session['id']}").status_code == 200
    assert client.delete(f"/sessions/{session['id']}").status_code == 404
