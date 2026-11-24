"""Tests for the hotel_booking_api Flask app, using Flask's test client."""
import pytest

from hotel_booking_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _room(client, number="101", room_type="double", rate=120, capacity=2):
    return client.post(
        "/rooms",
        json={
            "number": number,
            "type": room_type,
            "nightly_rate": rate,
            "capacity": capacity,
        },
    )


def _book(client, room_id=1, guest="Alice", check_in="2026-12-10",
          check_out="2026-12-13", guests=2):
    return client.post(
        "/bookings",
        json={
            "room_id": room_id,
            "guest_name": guest,
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests,
        },
    )


def test_create_room_returns_201(client):
    resp = _room(client)
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["id"] == 1
    assert data["number"] == "101"
    assert data["nightly_rate"] == 120


def test_create_room_duplicate_number_returns_409(client):
    _room(client)
    resp = _room(client, room_type="suite")
    assert resp.status_code == 409
    assert "already exists" in resp.get_json()["error"]


def test_create_room_invalid_rate_returns_400(client):
    assert _room(client, rate=0).status_code == 400
    assert _room(client, rate=-10).status_code == 400


def test_create_room_invalid_capacity_returns_400(client):
    assert _room(client, capacity=0).status_code == 400
    assert _room(client, capacity="two").status_code == 400


def test_create_room_missing_type_returns_400(client):
    resp = client.post(
        "/rooms", json={"number": "103", "nightly_rate": 100, "capacity": 2}
    )
    assert resp.status_code == 400


def test_list_rooms_and_get_room(client):
    _room(client)
    _room(client, number="102", room_type="suite", rate=250, capacity=4)
    rooms = client.get("/rooms").get_json()
    assert [r["number"] for r in rooms] == ["101", "102"]
    assert client.get("/rooms/2").get_json()["type"] == "suite"
    assert client.get("/rooms/99").status_code == 404


def test_room_available_before_any_booking(client):
    room = _room(client).get_json()
    resp = client.get(
        f"/rooms/{room['id']}/availability",
        query_string={"check_in": "2026-12-10", "check_out": "2026-12-13"},
    )
    assert resp.status_code == 200
    assert resp.get_json()["available"] is True


def test_create_booking_computes_nights_and_total(client):
    _room(client)
    resp = _book(client)
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["nights"] == 3
    assert data["total"] == 360
    assert data["status"] == "confirmed"


def test_overlapping_booking_returns_409(client):
    _room(client)
    _book(client)
    resp = _book(client, guest="Bob", check_in="2026-12-12", check_out="2026-12-15")
    assert resp.status_code == 409
    assert "already booked" in resp.get_json()["error"]


def test_adjacent_booking_after_checkout_is_allowed(client):
    _room(client)
    _book(client)
    resp = _book(client, guest="Bob", check_in="2026-12-13", check_out="2026-12-15")
    assert resp.status_code == 201
    assert resp.get_json()["nights"] == 2


def test_booking_unknown_room_returns_404(client):
    assert _book(client, room_id=99).status_code == 404


def test_booking_bad_date_format_returns_400(client):
    _room(client)
    assert _book(client, check_in="10/12/2026").status_code == 400
    assert _book(client, check_in="2026-13-01").status_code == 400


def test_booking_checkout_not_after_checkin_returns_400(client):
    _room(client)
    assert _book(client, check_in="2026-12-13", check_out="2026-12-13").status_code == 400
    assert _book(client, check_in="2026-12-14", check_out="2026-12-13").status_code == 400


def test_booking_too_many_guests_returns_400(client):
    _room(client)
    resp = _book(client, guests=3)
    assert resp.status_code == 400
    assert "fits 2 guests" in resp.get_json()["error"]


def test_booking_missing_guest_name_returns_400(client):
    _room(client)
    resp = client.post(
        "/bookings",
        json={"room_id": 1, "check_in": "2026-12-10",
              "check_out": "2026-12-12", "guests": 1},
    )
    assert resp.status_code == 400


def test_availability_endpoint_lists_only_free_rooms(client):
    _room(client)
    _room(client, number="102", room_type="suite", rate=250, capacity=4)
    _book(client)
    free = client.get(
        "/availability",
        query_string={"check_in": "2026-12-11", "check_out": "2026-12-12"},
    ).get_json()
    assert [r["number"] for r in free] == ["102"]


def test_availability_missing_dates_returns_400(client):
    _room(client)
    assert client.get("/availability").status_code == 400


def test_cancel_booking_frees_the_room(client):
    _room(client)
    booking = _book(client).get_json()
    resp = client.post(f"/bookings/{booking['id']}/cancel")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "cancelled"
    rebook = _book(client, guest="Bob", check_in="2026-12-11", check_out="2026-12-12")
    assert rebook.status_code == 201


def test_cancel_already_cancelled_booking_returns_400(client):
    _room(client)
    booking = _book(client).get_json()
    client.post(f"/bookings/{booking['id']}/cancel")
    assert client.post(f"/bookings/{booking['id']}/cancel").status_code == 400


def test_list_bookings_filters_by_room_and_status(client):
    _room(client)
    _room(client, number="102", room_type="suite", rate=250, capacity=4)
    first = _book(client).get_json()
    _book(client, room_id=2, guest="Bob")
    client.post(f"/bookings/{first['id']}/cancel")
    assert len(client.get("/bookings").get_json()) == 2
    assert len(client.get("/bookings", query_string={"room_id": 2}).get_json()) == 1
    cancelled = client.get("/bookings", query_string={"status": "cancelled"}).get_json()
    assert [b["id"] for b in cancelled] == [first["id"]]


def test_get_booking_and_missing_booking_404(client):
    _room(client)
    booking = _book(client).get_json()
    assert client.get(f"/bookings/{booking['id']}").get_json()["total"] == 360
    assert client.get("/bookings/99").status_code == 404
