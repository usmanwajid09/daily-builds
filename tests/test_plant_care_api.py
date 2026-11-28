"""Tests for the plant care API."""
from datetime import date, timedelta

import pytest

from plant_care_api.app import create_app


@pytest.fixture
def client():
    return create_app().test_client()


def _date(days_ago):
    return (date.today() - timedelta(days=days_ago)).isoformat()


def _create(client, **overrides):
    payload = {
        "name": "Monstera",
        "species": "Monstera deliciosa",
        "watering_interval_days": 7,
        "last_watered": _date(3),
    }
    payload.update(overrides)
    return client.post("/plants", json=payload)


def test_empty_list(client):
    response = client.get("/plants")
    assert response.status_code == 200
    assert response.get_json() == []


def test_create_201_with_computed_next_watering(client):
    response = _create(client)
    assert response.status_code == 201
    plant = response.get_json()
    assert plant["name"] == "Monstera"
    assert plant["species"] == "Monstera deliciosa"
    assert plant["last_watered"] == _date(3)
    expected_next = (date.today() + timedelta(days=4)).isoformat()
    assert plant["next_watering"] == expected_next
    assert plant["days_until_watering"] == 4


def test_create_defaults_last_watered_to_today(client):
    response = client.post(
        "/plants", json={"name": "Fern", "watering_interval_days": 3}
    )
    assert response.status_code == 201
    plant = response.get_json()
    assert plant["last_watered"] == date.today().isoformat()
    assert plant["species"] is None


def test_create_invalid_interval_400(client):
    for bad_interval in [0, -3, "weekly", None]:
        response = _create(client, watering_interval_days=bad_interval)
        assert response.status_code == 400
        assert "error" in response.get_json()


def test_create_missing_name_400(client):
    response = client.post("/plants", json={"watering_interval_days": 7})
    assert response.status_code == 400
    assert "error" in response.get_json()
    empty = client.post(
        "/plants", json={"name": "   ", "watering_interval_days": 7}
    )
    assert empty.status_code == 400


def test_create_bad_date_400(client):
    response = _create(client, last_watered="2026-13-40")
    assert response.status_code == 400
    assert "error" in response.get_json()
    response = _create(client, last_watered="yesterday")
    assert response.status_code == 400


def test_get_one(client):
    created = _create(client).get_json()
    response = client.get(f"/plants/{created['id']}")
    assert response.status_code == 200
    assert response.get_json()["name"] == "Monstera"


def test_get_unknown_404(client):
    response = client.get("/plants/999")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_update_interval_changes_next_watering(client):
    created = _create(client).get_json()
    response = client.put(
        f"/plants/{created['id']}", json={"watering_interval_days": 10}
    )
    assert response.status_code == 200
    updated = response.get_json()
    expected_next = (date.today() + timedelta(days=7)).isoformat()
    assert updated["next_watering"] == expected_next
    assert updated["days_until_watering"] == 7


def test_update_unknown_404_and_invalid_400(client):
    response = client.put("/plants/999", json={"name": "Fern"})
    assert response.status_code == 404
    created = _create(client).get_json()
    response = client.put(
        f"/plants/{created['id']}", json={"watering_interval_days": 0}
    )
    assert response.status_code == 400
    response = client.put(
        f"/plants/{created['id']}", json={"last_watered": "not-a-date"}
    )
    assert response.status_code == 400


def test_delete(client):
    created = _create(client).get_json()
    response = client.delete(f"/plants/{created['id']}")
    assert response.status_code == 200
    assert client.get(f"/plants/{created['id']}").status_code == 404
    assert client.get("/plants").get_json() == []


def test_delete_unknown_404(client):
    assert client.delete("/plants/999").status_code == 404


def test_due_list_includes_overdue_and_excludes_not_due(client):
    _create(client, name="Overdue", watering_interval_days=7, last_watered=_date(10))
    _create(client, name="DueToday", watering_interval_days=7, last_watered=_date(7))
    _create(client, name="NotDue", watering_interval_days=7, last_watered=_date(1))
    due = client.get("/plants/due").get_json()
    names = [p["name"] for p in due]
    assert names == ["Overdue", "DueToday"]
    assert "NotDue" not in names


def test_watering_updates_last_watered_and_next_watering(client):
    created = _create(client).get_json()
    response = client.post(f"/plants/{created['id']}/water", json={})
    assert response.status_code == 200
    watered = response.get_json()
    assert watered["last_watered"] == date.today().isoformat()
    expected_next = (date.today() + timedelta(days=7)).isoformat()
    assert watered["next_watering"] == expected_next
    assert watered["days_until_watering"] == 7


def test_water_log_records_entries_newest_first(client):
    created = _create(client).get_json()
    plant_id = created["id"]
    client.post(f"/plants/{plant_id}/water", json={"date": _date(2)})
    client.post(f"/plants/{plant_id}/water", json={})
    log = client.get(f"/plants/{plant_id}/water-log").get_json()
    dates = [entry["date"] for entry in log]
    assert dates == [date.today().isoformat(), _date(2), _date(3)]


def test_watering_unknown_plant_404(client):
    response = client.post("/plants/999/water", json={})
    assert response.status_code == 404
    assert "error" in response.get_json()
    assert client.get("/plants/999/water-log").status_code == 404


def test_watering_future_date_400(client):
    created = _create(client).get_json()
    future = (date.today() + timedelta(days=1)).isoformat()
    response = client.post(
        f"/plants/{created['id']}/water", json={"date": future}
    )
    assert response.status_code == 400
    assert "error" in response.get_json()
    bad = client.post(
        f"/plants/{created['id']}/water", json={"date": "not-a-date"}
    )
    assert bad.status_code == 400
