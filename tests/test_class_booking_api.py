"""Tests for the class_booking_api Flask gym class booking API."""

import pytest

from class_booking_api.app import create_app


@pytest.fixture
def client():
    return create_app().test_client()


def _make_class(client, capacity=2, **overrides):
    payload = {
        "name": "Morning Yoga",
        "instructor": "Sara",
        "date": "2026-11-29",
        "start_time": "08:00",
        "capacity": capacity,
    }
    payload.update(overrides)
    return client.post("/classes", json=payload)


def _book(client, class_id, member):
    return client.post(f"/classes/{class_id}/book", json={"member_name": member})


def test_list_empty(client):
    resp = client.get("/classes")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_create_class_201_with_spots_left(client):
    resp = _make_class(client, capacity=3)
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["name"] == "Morning Yoga"
    assert data["instructor"] == "Sara"
    assert data["capacity"] == 3
    assert data["spots_left"] == 3
    assert data["booked_count"] == 0
    assert data["waitlist_count"] == 0


def test_create_invalid_capacity_400(client):
    for bad in [0, -1, "2", 1.5, None]:
        resp = _make_class(client, capacity=bad)
        assert resp.status_code == 400, bad
        assert "error" in resp.get_json()


def test_create_bad_date_400(client):
    for bad_date in ["2026-13-01", "2026-02-30", "29-11-2026", "2026/11/29", ""]:
        resp = _make_class(client, date=bad_date)
        assert resp.status_code == 400, bad_date


def test_create_bad_time_and_missing_name_400(client):
    assert _make_class(client, start_time="25:00").status_code == 400
    assert _make_class(client, start_time="8am").status_code == 400
    assert _make_class(client, name="").status_code == 400
    assert _make_class(client, name="   ").status_code == 400
    resp = client.post("/classes", json={
        "date": "2026-11-29", "start_time": "08:00", "capacity": 2,
    })
    assert resp.status_code == 400


def test_get_unknown_class_404(client):
    resp = client.get("/classes/999")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_get_one_class(client):
    class_id = _make_class(client).get_json()["id"]
    resp = client.get(f"/classes/{class_id}")
    assert resp.status_code == 200
    assert resp.get_json()["spots_left"] == 2


def test_book_201_booked(client):
    class_id = _make_class(client).get_json()["id"]
    resp = _book(client, class_id, "Ali")
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["status"] == "booked"
    assert data["member_name"] == "Ali"
    assert client.get(f"/classes/{class_id}").get_json()["spots_left"] == 1


def test_book_when_full_waitlisted_with_position(client):
    class_id = _make_class(client, capacity=2).get_json()["id"]
    _book(client, class_id, "Ali")
    _book(client, class_id, "Bilal")
    resp = _book(client, class_id, "Usman")
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["status"] == "waitlisted"
    assert data["waitlist_position"] == 1
    resp2 = _book(client, class_id, "Hassan")
    assert resp2.get_json()["waitlist_position"] == 2


def test_duplicate_booking_409(client):
    class_id = _make_class(client).get_json()["id"]
    _book(client, class_id, "Ali")
    resp = _book(client, class_id, "Ali")
    assert resp.status_code == 409
    assert "error" in resp.get_json()


def test_duplicate_waitlist_409(client):
    class_id = _make_class(client, capacity=1).get_json()["id"]
    _book(client, class_id, "Ali")
    _book(client, class_id, "Bilal")
    resp = _book(client, class_id, "Bilal")
    assert resp.status_code == 409


def test_book_missing_member_name_400_and_unknown_class_404(client):
    class_id = _make_class(client).get_json()["id"]
    assert client.post(f"/classes/{class_id}/book", json={"member_name": ""}).status_code == 400
    assert client.post("/classes/999/book", json={"member_name": "Ali"}).status_code == 404


def test_cancel_booked_promotes_first_waitlisted(client):
    class_id = _make_class(client, capacity=2).get_json()["id"]
    _book(client, class_id, "Ali")
    _book(client, class_id, "Bilal")
    _book(client, class_id, "Usman")
    _book(client, class_id, "Hassan")
    resp = client.post(f"/classes/{class_id}/cancel", json={"member_name": "Ali"})
    assert resp.status_code == 200
    assert resp.get_json()["promoted"] == "Usman"
    bookings = client.get(f"/classes/{class_id}/bookings").get_json()
    assert [b["member_name"] for b in bookings] == ["Bilal", "Usman"]
    waitlist = client.get(f"/classes/{class_id}/waitlist").get_json()
    assert [w["member_name"] for w in waitlist] == ["Hassan"]


def test_cancel_waitlisted_member(client):
    class_id = _make_class(client, capacity=1).get_json()["id"]
    _book(client, class_id, "Ali")
    _book(client, class_id, "Bilal")
    resp = client.post(f"/classes/{class_id}/cancel", json={"member_name": "Bilal"})
    assert resp.status_code == 200
    assert resp.get_json()["promoted"] is None
    assert client.get(f"/classes/{class_id}/waitlist").get_json() == []
    assert [b["member_name"] for b in client.get(f"/classes/{class_id}/bookings").get_json()] == ["Ali"]


def test_cancel_unknown_member_404(client):
    class_id = _make_class(client).get_json()["id"]
    _book(client, class_id, "Ali")
    resp = client.post(f"/classes/{class_id}/cancel", json={"member_name": "Nobody"})
    assert resp.status_code == 404
    assert client.post("/classes/999/cancel", json={"member_name": "Ali"}).status_code == 404
    assert client.post(f"/classes/{class_id}/cancel", json={"member_name": ""}).status_code == 400


def test_bookings_and_waitlist_listings(client):
    class_id = _make_class(client, capacity=2).get_json()["id"]
    _book(client, class_id, "Ali")
    _book(client, class_id, "Bilal")
    _book(client, class_id, "Usman")
    _book(client, class_id, "Hassan")
    bookings = client.get(f"/classes/{class_id}/bookings").get_json()
    assert bookings == [
        {"member_name": "Ali", "status": "booked"},
        {"member_name": "Bilal", "status": "booked"},
    ]
    waitlist = client.get(f"/classes/{class_id}/waitlist").get_json()
    assert waitlist == [
        {"member_name": "Usman", "position": 1},
        {"member_name": "Hassan", "position": 2},
    ]
    assert client.get("/classes/999/bookings").status_code == 404
    assert client.get("/classes/999/waitlist").status_code == 404


def test_delete_class(client):
    class_id = _make_class(client).get_json()["id"]
    resp = client.delete(f"/classes/{class_id}")
    assert resp.status_code == 200
    assert client.get(f"/classes/{class_id}").status_code == 404
    assert client.delete("/classes/999").status_code == 404
