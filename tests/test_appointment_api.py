"""Tests for the appointment_api Flask app, using Flask's test client."""
from datetime import date, timedelta

import pytest

from appointment_api.app import create_app

TOMORROW = (date.today() + timedelta(days=1)).isoformat()
DAY_AFTER = (date.today() + timedelta(days=2)).isoformat()
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def service(client):
    return client.post(
        "/services",
        json={"name": "Haircut", "duration_minutes": 30, "price": 1500},
    ).get_json()


def _book(client, service_id, name="Usman", day=TOMORROW, start="10:00"):
    return client.post(
        "/appointments",
        json={"service_id": service_id, "customer_name": name,
              "date": day, "start_time": start},
    )


def test_list_services_starts_empty(client):
    assert client.get("/services").get_json() == []


def test_create_service_returns_201(client):
    resp = client.post(
        "/services",
        json={"name": "Haircut", "duration_minutes": 30, "price": 1500},
    )
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1, "name": "Haircut", "duration_minutes": 30, "price": 1500,
    }


def test_create_service_defaults_price_zero(client):
    service = client.post(
        "/services", json={"name": "Consult", "duration_minutes": 60},
    ).get_json()
    assert service["price"] == 0


def test_create_service_validation(client):
    assert client.post("/services", json={"name": ""}).status_code == 400
    assert client.post("/services", json={"name": "X"}).status_code == 400
    assert client.post(
        "/services", json={"name": "X", "duration_minutes": 0}).status_code == 400
    assert client.post(
        "/services", json={"name": "X", "duration_minutes": -30}
    ).status_code == 400
    assert client.post(
        "/services", json={"name": "X", "duration_minutes": "30"}
    ).status_code == 400
    assert client.post(
        "/services", json={"name": "X", "duration_minutes": 30, "price": -1}
    ).status_code == 400


def test_get_service_and_404(client, service):
    assert client.get(f"/services/{service['id']}").get_json()["name"] == "Haircut"
    assert client.get("/services/999").status_code == 404


def test_book_appointment_computes_end_time(client, service):
    resp = _book(client, service["id"])
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1, "service_id": service["id"], "customer_name": "Usman",
        "date": TOMORROW, "start_time": "10:00", "end_time": "10:30",
        "status": "booked",
    }


def test_book_validation_errors(client, service):
    assert _book(client, 999).status_code == 404
    assert _book(client, "x").status_code == 400
    assert _book(client, service["id"], name="  ").status_code == 400
    assert _book(client, service["id"], day="18-11-2026").status_code == 400
    assert _book(client, service["id"], day="2026-13-40").status_code == 400
    assert _book(client, service["id"], start="10am").status_code == 400
    assert _book(client, service["id"], start="25:00").status_code == 400


def test_double_booking_is_rejected(client, service):
    assert _book(client, service["id"], start="10:00").status_code == 201
    clash = _book(client, service["id"], name="Ali", start="10:15")
    assert clash.status_code == 409
    assert "overlaps" in clash.get_json()["error"]
    assert _book(client, service["id"], name="Ali", start="09:45").status_code == 409
    assert _book(client, service["id"], name="Ali", start="10:00").status_code == 409


def test_back_to_back_slots_are_allowed(client, service):
    _book(client, service["id"], start="10:00")
    assert _book(client, service["id"], name="Sara", start="10:30").status_code == 201
    assert _book(client, service["id"], name="Omar", start="09:30").status_code == 201


def test_same_time_different_service_or_day_is_allowed(client, service):
    other = client.post(
        "/services", json={"name": "Dental", "duration_minutes": 45},
    ).get_json()
    _book(client, service["id"], start="10:00")
    assert _book(client, other["id"], start="10:00").status_code == 201
    assert _book(client, service["id"], day=DAY_AFTER, start="10:00"
                  ).status_code == 201


def test_cancel_frees_the_slot(client, service):
    appt = _book(client, service["id"]).get_json()
    cancelled = client.post(f"/appointments/{appt['id']}/cancel").get_json()
    assert cancelled["status"] == "cancelled"
    assert _book(client, service["id"], name="Ali").status_code == 201
    assert client.post(f"/appointments/{appt['id']}/cancel").status_code == 400
    assert client.post("/appointments/999/cancel").status_code == 404


def test_upcoming_lists_future_booked_only(client, service):
    _book(client, service["id"], day=YESTERDAY, start="10:00")
    future = _book(client, service["id"], day=DAY_AFTER, start="09:00").get_json()
    cancelled = _book(client, service["id"], day=TOMORROW, start="10:00").get_json()
    client.post(f"/appointments/{cancelled['id']}/cancel")
    upcoming = client.get("/appointments/upcoming").get_json()
    assert [a["id"] for a in upcoming] == [future["id"]]


def test_upcoming_is_sorted(client, service):
    _book(client, service["id"], day=DAY_AFTER, start="11:00")
    _book(client, service["id"], day=TOMORROW, start="09:00")
    _book(client, service["id"], day=TOMORROW, start="08:00")
    upcoming = client.get("/appointments/upcoming").get_json()
    assert [(a["date"], a["start_time"]) for a in upcoming] == [
        (TOMORROW, "08:00"), (TOMORROW, "09:00"), (DAY_AFTER, "11:00"),
    ]


def test_list_filters(client, service):
    a = _book(client, service["id"]).get_json()
    b = _book(client, service["id"], day=DAY_AFTER, start="09:00").get_json()
    client.post(f"/appointments/{b['id']}/cancel")
    assert [x["id"] for x in client.get(
        "/appointments", query_string={"date": TOMORROW}).get_json()] == [a["id"]]
    assert [x["id"] for x in client.get(
        "/appointments", query_string={"status": "cancelled"}).get_json()] == [b["id"]]
    assert [x["id"] for x in client.get(
        "/appointments", query_string={"service_id": service["id"]}
    ).get_json()] == [a["id"], b["id"]]
    assert client.get("/appointments", query_string={"date": "bad"}
                      ).status_code == 400
    assert client.get("/appointments", query_string={"status": "nope"}
                      ).status_code == 400


def test_get_and_delete_appointment(client, service):
    appt = _book(client, service["id"]).get_json()
    assert client.get(f"/appointments/{appt['id']}").get_json()["id"] == appt["id"]
    assert client.get("/appointments/999").status_code == 404
    assert client.delete(f"/appointments/{appt['id']}").get_json() == {
        "deleted": appt["id"]
    }
    assert client.get(f"/appointments/{appt['id']}").status_code == 404
    assert client.delete(f"/appointments/{appt['id']}").status_code == 404


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
