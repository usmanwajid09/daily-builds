"""Tests for the car rental API."""
import pytest

from car_rental_api.app import create_app


@pytest.fixture()
def client():
    return create_app().test_client()


def _car(client, model="Corolla", rate=50):
    resp = client.post("/cars", json={
        "make": "Toyota", "model": model, "year": 2023, "daily_rate": rate,
    })
    assert resp.status_code == 201
    return resp.get_json()


def _book(client, car_id, start="2026-12-01", end="2026-12-03", customer="Usman"):
    return client.post("/bookings", json={
        "car_id": car_id, "customer_name": customer,
        "start_date": start, "end_date": end,
    })


def test_create_car(client):
    car = _car(client)
    assert car["make"] == "Toyota" and car["daily_rate"] == 50 and car["id"] == 1


def test_create_car_bad_rate(client):
    resp = client.post("/cars", json={
        "make": "Toyota", "model": "Corolla", "year": 2023, "daily_rate": 0,
    })
    assert resp.status_code == 400


def test_create_car_bad_year(client):
    resp = client.post("/cars", json={
        "make": "Toyota", "model": "Corolla", "year": 1800, "daily_rate": 50,
    })
    assert resp.status_code == 400


def test_get_and_list_cars(client):
    car = _car(client)
    assert client.get(f"/cars/{car['id']}").get_json()["model"] == "Corolla"
    assert len(client.get("/cars").get_json()) == 1
    assert client.get("/cars/99").status_code == 404


def test_booking_total_price(client):
    car = _car(client, rate=50)
    resp = _book(client, car["id"])
    assert resp.status_code == 201
    booking = resp.get_json()
    assert booking["days"] == 3
    assert booking["total_price"] == 150
    assert booking["status"] == "booked"


def test_single_day_booking(client):
    car = _car(client, rate=65)
    booking = _book(client, car["id"], "2026-12-05", "2026-12-05").get_json()
    assert booking["days"] == 1 and booking["total_price"] == 65


def test_double_booking_rejected(client):
    car = _car(client)
    assert _book(client, car["id"]).status_code == 201
    assert _book(client, car["id"], "2026-12-02", "2026-12-04", "Ali").status_code == 409
    assert _book(client, car["id"], "2026-11-30", "2026-12-01", "Ali").status_code == 409


def test_adjacent_booking_allowed(client):
    car = _car(client)
    assert _book(client, car["id"]).status_code == 201
    assert _book(client, car["id"], "2026-12-04", "2026-12-05", "Ali").status_code == 201


def test_booking_unknown_car_404(client):
    assert _book(client, 99).status_code == 404


def test_booking_bad_dates_400(client):
    car = _car(client)
    assert _book(client, car["id"], "2026-12-03", "2026-12-01").status_code == 400
    assert _book(client, car["id"], "not-a-date", "2026-12-03").status_code == 400


def test_availability(client):
    car = _car(client)
    other = _car(client, model="Civic", rate=65)
    _book(client, car["id"])
    free = client.get("/cars/available?start_date=2026-12-02&end_date=2026-12-04").get_json()
    assert [c["id"] for c in free] == [other["id"]]
    assert client.get("/cars/available?start_date=bad&end_date=2026-12-04").status_code == 400


def test_cancel_frees_car(client):
    car = _car(client)
    booking = _book(client, car["id"]).get_json()
    assert client.post(f"/bookings/{booking['id']}/cancel").status_code == 200
    assert _book(client, car["id"], "2026-12-02", "2026-12-04", "Ali").status_code == 201
    assert client.post(f"/bookings/{booking['id']}/cancel").status_code == 409


def test_revenue(client):
    car = _car(client, rate=50)
    _book(client, car["id"])
    _book(client, car["id"], "2026-12-10", "2026-12-11", "Ali")
    revenue = client.get(f"/cars/{car['id']}/revenue").get_json()
    assert revenue["booking_count"] == 2
    assert revenue["total_revenue"] == 250


def test_get_booking_404(client):
    assert client.get("/bookings/99").status_code == 404
