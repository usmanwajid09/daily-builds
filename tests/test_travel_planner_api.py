"""Tests for the travel planner API."""
import pytest

from travel_planner_api.app import create_app


@pytest.fixture()
def client():
    return create_app().test_client()


def _trip(client):
    resp = client.post("/trips", json={
        "name": "Northern Areas", "destination": "Hunza",
        "start_date": "2026-12-10", "end_date": "2026-12-12",
    })
    assert resp.status_code == 201
    return resp.get_json()


def _day(client, trip_id, date="2026-12-10", title="Arrival"):
    resp = client.post(f"/trips/{trip_id}/days", json={"date": date, "title": title})
    assert resp.status_code == 201
    return resp.get_json()


def _activity(client, trip_id, day_id, name="Hotel", cost=8000):
    resp = client.post(
        f"/trips/{trip_id}/days/{day_id}/activities",
        json={"name": name, "cost": cost},
    )
    assert resp.status_code == 201
    return resp.get_json()


def test_create_trip(client):
    trip = _trip(client)
    assert trip["destination"] == "Hunza"
    assert trip["total_cost"] == 0
    assert trip["day_count"] == 0


def test_create_trip_bad_dates(client):
    resp = client.post("/trips", json={
        "name": "X", "destination": "Y",
        "start_date": "2026-12-12", "end_date": "2026-12-10",
    })
    assert resp.status_code == 400
    resp = client.post("/trips", json={
        "name": "X", "destination": "Y",
        "start_date": "not-a-date", "end_date": "2026-12-10",
    })
    assert resp.status_code == 400


def test_add_day_and_activity(client):
    trip = _trip(client)
    day = _day(client, trip["id"])
    activity = _activity(client, trip["id"], day["id"])
    assert activity["name"] == "Hotel" and activity["cost"] == 8000
    fetched = client.get(f"/trips/{trip['id']}").get_json()
    assert fetched["day_count"] == 1 and fetched["activity_count"] == 1
    assert fetched["total_cost"] == 8000


def test_day_outside_trip_400(client):
    trip = _trip(client)
    resp = client.post(f"/trips/{trip['id']}/days", json={"date": "2026-12-20", "title": "Too late"})
    assert resp.status_code == 400


def test_duplicate_day_409(client):
    trip = _trip(client)
    _day(client, trip["id"])
    resp = client.post(f"/trips/{trip['id']}/days", json={"date": "2026-12-10", "title": "Again"})
    assert resp.status_code == 409


def test_day_unknown_trip_404(client):
    resp = client.post("/trips/99/days", json={"date": "2026-12-10", "title": "X"})
    assert resp.status_code == 404


def test_activity_negative_cost_400(client):
    trip = _trip(client)
    day = _day(client, trip["id"])
    resp = client.post(
        f"/trips/{trip['id']}/days/{day['id']}/activities",
        json={"name": "Bad", "cost": -100},
    )
    assert resp.status_code == 400


def test_activity_unknown_day_404(client):
    trip = _trip(client)
    resp = client.post(
        f"/trips/{trip['id']}/days/99/activities",
        json={"name": "Hotel", "cost": 100},
    )
    assert resp.status_code == 404


def test_cost_breakdown(client):
    trip = _trip(client)
    day1 = _day(client, trip["id"], "2026-12-10", "Arrival")
    day2 = _day(client, trip["id"], "2026-12-11", "Lake day")
    _activity(client, trip["id"], day1["id"], "Hotel", 8000)
    _activity(client, trip["id"], day1["id"], "Dinner", 2500)
    _activity(client, trip["id"], day2["id"], "Boat ride", 3000)
    _activity(client, trip["id"], day2["id"], "Jeep hire", 6000)
    cost = client.get(f"/trips/{trip['id']}/cost").get_json()
    assert cost["total_cost"] == 19500
    assert cost["by_day"][0]["cost"] == 10500
    assert cost["by_day"][1]["cost"] == 9000
    assert cost["by_day"][0]["activity_count"] == 2


def test_list_days_sorted(client):
    trip = _trip(client)
    _day(client, trip["id"], "2026-12-11", "Second")
    _day(client, trip["id"], "2026-12-10", "First")
    days = client.get(f"/trips/{trip['id']}/days").get_json()
    assert [d["date"] for d in days] == ["2026-12-10", "2026-12-11"]


def test_delete_day_removes_activities(client):
    trip = _trip(client)
    day = _day(client, trip["id"])
    _activity(client, trip["id"], day["id"])
    assert client.delete(f"/trips/{trip['id']}/days/{day['id']}").status_code == 200
    assert client.get(f"/trips/{trip['id']}").get_json()["total_cost"] == 0


def test_delete_trip(client):
    trip = _trip(client)
    day = _day(client, trip["id"])
    _activity(client, trip["id"], day["id"])
    assert client.delete(f"/trips/{trip['id']}").status_code == 200
    assert client.get(f"/trips/{trip['id']}").status_code == 404
    assert client.get(f"/trips/{trip['id']}/cost").status_code == 404


def test_get_trip_404(client):
    assert client.get("/trips/99").status_code == 404
