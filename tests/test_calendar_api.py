"""Tests for calendar_api."""
import pytest

from calendar_api.app import create_app

EVENT = {
    "title": "Team standup",
    "start": "2026-11-17T09:00",
    "end": "2026-11-17T09:30",
    "location": "Room 4",
}


@pytest.fixture
def client():
    return create_app().test_client()


def _create(client, payload=None):
    return client.post("/events", json=payload if payload is not None else EVENT)


def test_create_event(client):
    resp = _create(client)
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["id"] == 1
    assert data["title"] == "Team standup"
    assert data["location"] == "Room 4"


def test_list_sorted_by_start(client):
    _create(client, {**EVENT, "title": "Later",
                     "start": "2026-11-17T14:00", "end": "2026-11-17T15:00"})
    _create(client)
    listed = client.get("/events").get_json()
    assert [e["title"] for e in listed] == ["Team standup", "Later"]


def test_filter_by_date(client):
    _create(client)
    _create(client, {**EVENT, "title": "Tomorrow",
                     "start": "2026-11-18T10:00", "end": "2026-11-18T11:00"})
    day = client.get("/events?date=2026-11-17").get_json()
    assert [e["title"] for e in day] == ["Team standup"]


def test_filter_by_date_bad_format_400(client):
    assert client.get("/events?date=17-11-2026").status_code == 400


def test_upcoming_excludes_past(client):
    _create(client, {**EVENT, "title": "Past",
                     "start": "2026-11-16T10:00", "end": "2026-11-16T11:00"})
    _create(client)
    upcoming = client.get("/events/upcoming?now=2026-11-17T00:00").get_json()
    assert [e["title"] for e in upcoming] == ["Team standup"]


def test_get_event(client):
    _create(client)
    assert client.get("/events/1").get_json()["title"] == "Team standup"


def test_update_event(client):
    _create(client)
    resp = client.put("/events/1", json={
        "title": "Standup moved", "start": "2026-11-17T10:00",
        "end": "2026-11-17T10:30"})
    assert resp.status_code == 200
    assert resp.get_json()["title"] == "Standup moved"
    assert resp.get_json()["start"] == "2026-11-17T10:00"


def test_update_end_before_start_400(client):
    _create(client)
    resp = client.put("/events/1", json={
        "title": "Bad", "start": "2026-11-17T10:00",
        "end": "2026-11-17T09:00"})
    assert resp.status_code == 400


def test_create_end_before_start_400(client):
    resp = _create(client, {**EVENT, "start": "2026-11-17T10:00",
                            "end": "2026-11-17T09:00"})
    assert resp.status_code == 400
    assert "end must be after start" in resp.get_json()["error"]


def test_create_end_equals_start_400(client):
    resp = _create(client, {**EVENT, "start": "2026-11-17T09:00",
                            "end": "2026-11-17T09:00"})
    assert resp.status_code == 400


def test_create_bad_timestamp_400(client):
    resp = _create(client, {**EVENT, "start": "tomorrow 9am"})
    assert resp.status_code == 400


def test_create_missing_title_400(client):
    resp = _create(client, {"start": "2026-11-17T09:00",
                            "end": "2026-11-17T09:30"})
    assert resp.status_code == 400


def test_unknown_event_404(client):
    assert client.get("/events/99").status_code == 404
    assert client.put("/events/99", json=EVENT).status_code == 404


def test_delete_event(client):
    _create(client)
    assert client.delete("/events/1").status_code == 200
    assert client.get("/events/1").status_code == 404
