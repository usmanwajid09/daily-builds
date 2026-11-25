"""Tests for the volunteer events API."""
import pytest

from volunteer_api.app import create_app


@pytest.fixture()
def client():
    return create_app().test_client()


def _event(client, capacity=2, name="Food Drive"):
    resp = client.post("/events", json={
        "name": name, "date": "2026-11-29", "capacity": capacity, "location": "Lahore",
    })
    assert resp.status_code == 201
    return resp.get_json()


def _signup(client, event_id, name="Usman"):
    resp = client.post(f"/events/{event_id}/signups", json={"volunteer_name": name})
    assert resp.status_code == 201
    return resp.get_json()


def test_create_event(client):
    event = _event(client)
    assert event["name"] == "Food Drive"
    assert event["capacity"] == 2
    assert event["spots_remaining"] == 2
    assert event["signup_count"] == 0


def test_create_event_bad_capacity(client):
    resp = client.post("/events", json={"name": "X", "date": "2026-11-29", "capacity": 0})
    assert resp.status_code == 400


def test_create_event_bad_date(client):
    resp = client.post("/events", json={"name": "X", "date": "29 Nov", "capacity": 5})
    assert resp.status_code == 400


def test_signup_success(client):
    event = _event(client)
    signup = _signup(client, event["id"])
    assert signup["volunteer_name"] == "Usman"
    assert signup["hours"] == 0
    fetched = client.get(f"/events/{event['id']}").get_json()
    assert fetched["signup_count"] == 1
    assert fetched["spots_remaining"] == 1


def test_over_capacity_409(client):
    event = _event(client, capacity=2)
    _signup(client, event["id"], "Usman")
    _signup(client, event["id"], "Ali")
    resp = client.post(f"/events/{event['id']}/signups", json={"volunteer_name": "Sara"})
    assert resp.status_code == 409


def test_duplicate_signup_409(client):
    event = _event(client, capacity=5)
    _signup(client, event["id"], "Usman")
    resp = client.post(f"/events/{event['id']}/signups", json={"volunteer_name": "usman"})
    assert resp.status_code == 409


def test_signup_missing_name_400(client):
    event = _event(client)
    assert client.post(f"/events/{event['id']}/signups", json={}).status_code == 400


def test_signup_unknown_event_404(client):
    assert client.post("/events/99/signups", json={"volunteer_name": "Usman"}).status_code == 404


def test_log_hours(client):
    event = _event(client)
    signup = _signup(client, event["id"])
    resp = client.post(
        f"/events/{event['id']}/signups/{signup['id']}/hours", json={"hours": 4},
    )
    assert resp.status_code == 201 or resp.status_code == 200
    assert resp.get_json()["hours"] == 4
    resp = client.post(
        f"/events/{event['id']}/signups/{signup['id']}/hours", json={"hours": 2.5},
    )
    assert resp.get_json()["hours"] == 6.5


def test_log_hours_bad_value(client):
    event = _event(client)
    signup = _signup(client, event["id"])
    url = f"/events/{event['id']}/signups/{signup['id']}/hours"
    assert client.post(url, json={"hours": 0}).status_code == 400
    assert client.post(url, json={"hours": -2}).status_code == 400


def test_log_hours_unknown_signup_404(client):
    event = _event(client)
    resp = client.post(f"/events/{event['id']}/signups/99/hours", json={"hours": 2})
    assert resp.status_code == 404


def test_volunteer_hours_total(client):
    event1 = _event(client, capacity=5, name="Food Drive")
    event2 = _event(client, capacity=5, name="Beach Cleanup")
    s1 = _signup(client, event1["id"], "Usman")
    s2 = _signup(client, event2["id"], "Usman")
    client.post(f"/events/{event1['id']}/signups/{s1['id']}/hours", json={"hours": 4})
    client.post(f"/events/{event2['id']}/signups/{s2['id']}/hours", json={"hours": 3})
    totals = client.get("/volunteers?name=Usman").get_json()
    assert totals["total_hours"] == 7
    assert totals["signup_count"] == 2
    assert totals["events"] == ["Food Drive", "Beach Cleanup"]


def test_volunteer_hours_missing_name(client):
    assert client.get("/volunteers").status_code == 400


def test_list_signups(client):
    event = _event(client)
    _signup(client, event["id"], "Usman")
    _signup(client, event["id"], "Ali")
    signups = client.get(f"/events/{event['id']}/signups").get_json()
    assert len(signups) == 2


def test_delete_event(client):
    event = _event(client)
    _signup(client, event["id"])
    assert client.delete(f"/events/{event['id']}").status_code == 200
    assert client.get(f"/events/{event['id']}").status_code == 404
