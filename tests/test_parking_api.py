"""Tests for the parking_api Flask app, using Flask's test client."""
import pytest

from parking_api.app import create_app

START = "2026-11-22T08:00:00+00:00"


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def spot(client):
    return client.post(
        "/spots",
        json={"spot_number": "A1", "zone": "Ground", "hourly_rate": 100},
    ).get_json()


def _check_in(client, spot_id, plate="LEA-123", started_at=START):
    return client.post(
        f"/spots/{spot_id}/check-in",
        json={"license_plate": plate, "started_at": started_at},
    )


def test_create_spot_returns_201_free(client):
    resp = client.post(
        "/spots", json={"spot_number": "A1", "zone": "Ground", "hourly_rate": 100}
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["spot_number"] == "A1"
    assert body["status"] == "free"
    assert body["vehicle_type"] == "car"
    assert body["hourly_rate"] == 100.0


def test_create_spot_defaults(client):
    resp = client.post("/spots", json={"spot_number": "B2"})
    assert resp.status_code == 201
    assert resp.get_json()["hourly_rate"] == 100.0
    assert resp.get_json()["zone"] is None


def test_create_spot_rejects_duplicate_number(client, spot):
    assert client.post("/spots", json={"spot_number": "a1"}).status_code == 409


def test_create_spot_rejects_bad_vehicle_type(client):
    assert client.post(
        "/spots", json={"spot_number": "A1", "vehicle_type": "bus"}
    ).status_code == 400


def test_create_spot_rejects_negative_rate(client):
    assert client.post(
        "/spots", json={"spot_number": "A1", "hourly_rate": -10}
    ).status_code == 400


def test_create_spot_rejects_missing_number(client):
    assert client.post("/spots", json={"zone": "Ground"}).status_code == 400


def test_get_spot_and_404(client, spot):
    assert client.get(f"/spots/{spot['id']}").get_json()["zone"] == "Ground"
    assert client.get("/spots/999").status_code == 404


def test_list_spots_filters(client, spot):
    client.post(
        "/spots",
        json={"spot_number": "B1", "zone": "Basement", "vehicle_type": "motorcycle"},
    )
    ground = client.get("/spots", query_string={"zone": "Ground"}).get_json()
    assert [s["spot_number"] for s in ground] == ["A1"]
    bikes = client.get("/spots", query_string={"vehicle_type": "motorcycle"}).get_json()
    assert [s["spot_number"] for s in bikes] == ["B1"]
    assert client.get("/spots", query_string={"vehicle_type": "bus"}).status_code == 400


def test_update_spot(client, spot):
    resp = client.put(f"/spots/{spot['id']}", json={"hourly_rate": 150, "zone": "VIP"})
    assert resp.status_code == 200
    assert resp.get_json()["hourly_rate"] == 150.0
    assert resp.get_json()["zone"] == "VIP"


def test_update_spot_rejects_duplicate_number(client, spot):
    client.post("/spots", json={"spot_number": "A2"})
    assert client.put(
        f"/spots/{spot['id']}", json={"spot_number": "A2"}
    ).status_code == 409


def test_delete_free_spot_without_sessions(client, spot):
    assert client.delete(f"/spots/{spot['id']}").status_code == 200
    assert client.get(f"/spots/{spot['id']}").status_code == 404


def test_delete_occupied_spot_returns_409(client, spot):
    _check_in(client, spot["id"])
    assert client.delete(f"/spots/{spot['id']}").status_code == 409


def test_delete_spot_with_sessions_returns_409(client, spot):
    _check_in(client, spot["id"])
    client.post("/spots/1/check-out", json={"ended_at": "2026-11-22T09:00:00+00:00"})
    assert client.delete(f"/spots/{spot['id']}").status_code == 409


def test_check_in_returns_201_active(client, spot):
    resp = _check_in(client, spot["id"])
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["license_plate"] == "LEA-123"
    assert body["status"] == "active"
    assert body["fee"] is None
    assert client.get(f"/spots/{spot['id']}").get_json()["status"] == "occupied"


def test_check_in_to_occupied_spot_returns_409(client, spot):
    _check_in(client, spot["id"])
    assert _check_in(client, spot["id"], plate="LEA-999").status_code == 409


def test_check_in_requires_plate(client, spot):
    resp = client.post(f"/spots/{spot['id']}/check-in", json={})
    assert resp.status_code == 400


def test_check_in_wrong_vehicle_type_returns_409(client):
    bike_spot = client.post(
        "/spots",
        json={"spot_number": "B1", "vehicle_type": "motorcycle"},
    ).get_json()
    resp = client.post(
        f"/spots/{bike_spot['id']}/check-in",
        json={"license_plate": "LEA-1", "vehicle_type": "car"},
    )
    assert resp.status_code == 409


def test_check_in_missing_spot_404(client):
    assert _check_in(client, 999).status_code == 404


def test_check_out_charges_per_started_hour(client, spot):
    _check_in(client, spot["id"])
    resp = client.post(
        "/spots/1/check-out", json={"ended_at": "2026-11-22T10:30:00+00:00"}
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["duration_minutes"] == 150.0
    assert body["hours_billed"] == 3
    assert body["fee"] == 300.0
    assert body["status"] == "completed"
    assert client.get("/spots/1").get_json()["status"] == "free"


def test_check_out_minimum_one_hour(client, spot):
    _check_in(client, spot["id"])
    resp = client.post(
        "/spots/1/check-out", json={"ended_at": "2026-11-22T08:20:00+00:00"}
    )
    assert resp.get_json()["hours_billed"] == 1
    assert resp.get_json()["fee"] == 100.0


def test_check_out_exact_hour_boundary(client, spot):
    _check_in(client, spot["id"])
    resp = client.post(
        "/spots/1/check-out", json={"ended_at": "2026-11-22T10:00:00+00:00"}
    )
    assert resp.get_json()["hours_billed"] == 2
    assert resp.get_json()["fee"] == 200.0


def test_check_out_free_spot_returns_409(client, spot):
    assert client.post("/spots/1/check-out", json={}).status_code == 409


def test_check_out_before_start_returns_400(client, spot):
    _check_in(client, spot["id"])
    assert client.post(
        "/spots/1/check-out", json={"ended_at": "2026-11-22T07:00:00+00:00"}
    ).status_code == 400


def test_check_out_bad_timestamp_returns_400(client, spot):
    _check_in(client, spot["id"])
    assert client.post(
        "/spots/1/check-out", json={"ended_at": "not a time"}
    ).status_code == 400


def test_summary_tracks_occupancy_and_revenue(client, spot):
    client.post("/spots", json={"spot_number": "A2"})
    client.post("/spots", json={"spot_number": "A3"})
    summary = client.get("/summary").get_json()
    assert summary == {
        "total_spots": 3,
        "occupied_spots": 0,
        "free_spots": 3,
        "occupancy_percent": 0.0,
        "total_revenue": 0.0,
    }
    _check_in(client, spot["id"])
    summary = client.get("/summary").get_json()
    assert summary["occupied_spots"] == 1
    assert summary["occupancy_percent"] == 33.33
    assert summary["total_revenue"] == 0.0
    client.post("/spots/1/check-out", json={"ended_at": "2026-11-22T09:00:00+00:00"})
    summary = client.get("/summary").get_json()
    assert summary["occupied_spots"] == 0
    assert summary["total_revenue"] == 100.0


def test_list_sessions_filters_by_status(client, spot):
    _check_in(client, spot["id"])
    active = client.get("/sessions", query_string={"status": "active"}).get_json()
    assert len(active) == 1
    client.post("/spots/1/check-out", json={"ended_at": "2026-11-22T09:00:00+00:00"})
    completed = client.get(
        "/sessions", query_string={"status": "completed"}
    ).get_json()
    assert len(completed) == 1
    assert client.get("/sessions/999").status_code == 404
    assert client.get("/sessions", query_string={"status": "bogus"}).status_code == 400


def test_list_spot_sessions(client, spot):
    session = _check_in(client, spot["id"]).get_json()
    sessions = client.get(f"/spots/{spot['id']}/sessions").get_json()
    assert [s["id"] for s in sessions] == [session["id"]]
    assert client.get("/spots/999/sessions").status_code == 404
