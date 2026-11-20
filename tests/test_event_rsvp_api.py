"""Tests for the event_rsvp_api Flask app, using Flask's test client."""
import pytest

from event_rsvp_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _event(client, title="Board game night", date="2026-11-27", capacity=2,
           **extra):
    payload = {"title": title, "date": date, "capacity": capacity}
    payload.update(extra)
    return client.post("/events", json=payload)


def _rsvp(client, event_id, name="Ali", response="yes"):
    return client.post(f"/events/{event_id}/rsvp",
                       json={"name": name, "response": response})


def test_list_starts_empty(client):
    assert client.get("/events").get_json() == []


def test_create_event_returns_201(client):
    resp = _event(client, location="Usman's place")
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "title": "Board game night",
        "date": "2026-11-27",
        "capacity": 2,
        "location": "Usman's place",
        "yes_count": 0,
        "no_count": 0,
        "maybe_count": 0,
        "spots_remaining": 2,
    }


def test_create_location_defaults_to_none(client):
    assert _event(client).get_json()["location"] is None


def test_create_missing_title_400(client):
    resp = client.post("/events", json={"date": "2026-11-27", "capacity": 5})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_create_bad_date_400(client):
    assert _event(client, date="27-11-2026").status_code == 400
    assert _event(client, date="2026-13-01").status_code == 400
    assert _event(client, date="tomorrow").status_code == 400


def test_create_bad_capacity_400(client):
    assert _event(client, capacity=0).status_code == 400
    assert _event(client, capacity=-3).status_code == 400
    assert _event(client, capacity="many").status_code == 400


def test_get_event_and_404(client):
    _event(client)
    assert client.get("/events/1").get_json()["title"] == "Board game night"
    assert client.get("/events/99").status_code == 404


def test_filter_by_date(client):
    _event(client)
    _event(client, title="Picnic", date="2026-11-28", capacity=5)
    found = client.get("/events", query_string={"date": "2026-11-27"}).get_json()
    assert [e["title"] for e in found] == ["Board game night"]
    assert client.get("/events", query_string={"date": "bad"}).status_code == 400


def test_rsvp_yes_returns_201(client):
    _event(client)
    resp = _rsvp(client, 1)
    assert resp.status_code == 201
    assert resp.get_json() == {"id": 1, "event_id": 1, "name": "Ali",
                               "response": "yes"}


def test_rsvp_bad_response_400(client):
    _event(client)
    assert _rsvp(client, 1, response="probably").status_code == 400


def test_rsvp_missing_name_400(client):
    _event(client)
    resp = client.post("/events/1/rsvp", json={"response": "yes"})
    assert resp.status_code == 400


def test_rsvp_unknown_event_404(client):
    assert _rsvp(client, 42).status_code == 404


def test_capacity_enforced_with_409(client):
    _event(client, capacity=2)
    _rsvp(client, 1, "Ali", "yes")
    _rsvp(client, 1, "Sara", "yes")
    resp = _rsvp(client, 1, "Bilal", "yes")
    assert resp.status_code == 409
    assert "error" in resp.get_json()


def test_maybe_and_no_do_not_take_spots(client):
    _event(client, capacity=1)
    _rsvp(client, 1, "Ali", "yes")
    assert _rsvp(client, 1, "Bilal", "maybe").status_code == 201
    assert _rsvp(client, 1, "Sara", "no").status_code == 201
    summary = client.get("/events/1/summary").get_json()
    assert summary["spots_remaining"] == 0
    assert summary["maybe"] == 1
    assert summary["no"] == 1


def test_repeat_rsvp_updates_instead_of_duplicating(client):
    _event(client, capacity=2)
    _rsvp(client, 1, "Ali", "maybe")
    resp = _rsvp(client, 1, "Ali", "yes")
    assert resp.status_code == 200
    assert resp.get_json()["id"] == 1
    assert len(client.get("/events/1/attendees").get_json()) == 1


def test_repeat_rsvp_to_yes_when_full_409(client):
    _event(client, capacity=1)
    _rsvp(client, 1, "Ali", "yes")
    _rsvp(client, 1, "Bilal", "maybe")
    assert _rsvp(client, 1, "Bilal", "yes").status_code == 409


def test_attendees_list_and_filter(client):
    _event(client)
    _rsvp(client, 1, "Ali", "yes")
    _rsvp(client, 1, "Sara", "maybe")
    all_rows = client.get("/events/1/attendees").get_json()
    assert [a["name"] for a in all_rows] == ["Ali", "Sara"]
    yes = client.get("/events/1/attendees",
                     query_string={"response": "yes"}).get_json()
    assert [a["name"] for a in yes] == ["Ali"]
    assert client.get("/events/1/attendees",
                      query_string={"response": "x"}).status_code == 400
    assert client.get("/events/9/attendees").status_code == 404


def test_update_rsvp_response(client):
    _event(client, capacity=1)
    _rsvp(client, 1, "Ali", "yes")
    resp = client.put("/rsvps/1", json={"response": "no"})
    assert resp.status_code == 200
    assert resp.get_json()["response"] == "no"
    assert client.get("/events/1/summary").get_json()["spots_remaining"] == 1


def test_update_rsvp_to_yes_when_full_409(client):
    _event(client, capacity=1)
    _rsvp(client, 1, "Ali", "yes")
    _rsvp(client, 1, "Bilal", "maybe")
    assert client.put("/rsvps/2", json={"response": "yes"}).status_code == 409


def test_update_rsvp_404_and_bad_response(client):
    assert client.put("/rsvps/5", json={"response": "yes"}).status_code == 404
    _event(client)
    _rsvp(client, 1, "Ali", "yes")
    assert client.put("/rsvps/1", json={"response": "eh"}).status_code == 400


def test_delete_rsvp_frees_a_spot(client):
    _event(client, capacity=1)
    _rsvp(client, 1, "Ali", "yes")
    assert client.delete("/rsvps/1").get_json() == {"deleted": 1}
    assert client.get("/events/1/summary").get_json()["spots_remaining"] == 1
    assert client.delete("/rsvps/1").status_code == 404


def test_summary(client):
    _event(client, capacity=5)
    _rsvp(client, 1, "Ali", "yes")
    _rsvp(client, 1, "Sara", "yes")
    _rsvp(client, 1, "Bilal", "maybe")
    _rsvp(client, 1, "Zara", "no")
    summary = client.get("/events/1/summary").get_json()
    assert summary == {
        "event_id": 1,
        "title": "Board game night",
        "capacity": 5,
        "yes": 2,
        "no": 1,
        "maybe": 1,
        "total_rsvps": 4,
        "spots_remaining": 3,
    }


def test_update_event(client):
    _event(client)
    resp = client.put("/events/1", json={"capacity": 10,
                                         "location": "Library hall"})
    assert resp.status_code == 200
    assert resp.get_json()["capacity"] == 10
    assert resp.get_json()["spots_remaining"] == 10


def test_update_event_capacity_below_yes_count_400(client):
    _event(client, capacity=3)
    _rsvp(client, 1, "Ali", "yes")
    _rsvp(client, 1, "Sara", "yes")
    assert client.put("/events/1", json={"capacity": 1}).status_code == 400


def test_delete_event_removes_rsvps(client):
    _event(client)
    _rsvp(client, 1, "Ali", "yes")
    assert client.delete("/events/1").get_json() == {"deleted": 1}
    assert client.get("/events").get_json() == []
    assert client.get("/events/1/attendees").status_code == 404
