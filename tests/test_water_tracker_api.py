"""Tests for water_tracker_api."""
import pytest

from water_tracker_api.app import create_app


@pytest.fixture()
def client():
    return create_app().test_client()


def test_default_target(client):
    assert client.get("/target").get_json() == {"target_ml": 2000}


def test_set_target(client):
    resp = client.put("/target", json={"target_ml": 3000})
    assert resp.status_code == 200
    assert resp.get_json()["target_ml"] == 3000


def test_set_target_invalid(client):
    assert client.put("/target", json={"target_ml": 0}).status_code == 400
    assert client.put("/target", json={"target_ml": -100}).status_code == 400
    assert client.put("/target", json={"target_ml": "lots"}).status_code == 400


def test_log_intake_ml(client):
    resp = client.post("/intake", json={"date": "2026-11-23", "amount_ml": 500})
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["amount_ml"] == 500
    assert body["day_total_ml"] == 500


def test_log_intake_glasses(client):
    resp = client.post("/intake", json={"date": "2026-11-23", "glasses": 4})
    assert resp.status_code == 201
    assert resp.get_json()["amount_ml"] == 1000  # 4 x 250 ml


def test_log_intake_accumulates(client):
    client.post("/intake", json={"date": "2026-11-23", "amount_ml": 500})
    client.post("/intake", json={"date": "2026-11-23", "glasses": 2})
    listed = client.get("/intake?date=2026-11-23").get_json()
    assert listed["total_ml"] == 1000
    assert len(listed["entries"]) == 2


def test_log_intake_validation(client):
    assert client.post("/intake", json={"amount_ml": 0}).status_code == 400
    assert client.post("/intake", json={"amount_ml": -50}).status_code == 400
    assert client.post("/intake", json={}).status_code == 400
    assert client.post("/intake", json={"date": "2026-13-01", "amount_ml": 100}).status_code == 400
    assert client.post("/intake", json={"glasses": 0}).status_code == 400


def test_summary_met_and_unmet(client):
    client.post("/intake", json={"date": "2026-11-23", "amount_ml": 500})
    summary = client.get("/summary/2026-11-23").get_json()
    assert summary["total_ml"] == 500
    assert summary["met_target"] is False
    assert summary["remaining_ml"] == 1500
    client.post("/intake", json={"date": "2026-11-23", "amount_ml": 1600})
    summary = client.get("/summary/2026-11-23").get_json()
    assert summary["met_target"] is True
    assert summary["remaining_ml"] == 0


def test_summary_empty_day(client):
    summary = client.get("/summary/2026-11-20").get_json()
    assert summary["total_ml"] == 0
    assert summary["met_target"] is False


def test_summary_bad_date(client):
    assert client.get("/summary/not-a-date").status_code == 400


def test_streak_empty(client):
    assert client.get("/streak").get_json()["streak_days"] == 0


def test_streak_counts_consecutive_days(client):
    for day in ("2026-11-21", "2026-11-22", "2026-11-23"):
        client.post("/intake", json={"date": day, "amount_ml": 2100})
    assert client.get("/streak").get_json()["streak_days"] == 3


def test_streak_breaks_on_missed_day(client):
    client.post("/intake", json={"date": "2026-11-21", "amount_ml": 2100})
    client.post("/intake", json={"date": "2026-11-22", "amount_ml": 500})
    client.post("/intake", json={"date": "2026-11-23", "amount_ml": 2100})
    assert client.get("/streak").get_json()["streak_days"] == 1


def test_streak_uses_target(client):
    client.put("/target", json={"target_ml": 3000})
    client.post("/intake", json={"date": "2026-11-23", "amount_ml": 2100})
    assert client.get("/streak").get_json()["streak_days"] == 0


def test_weekly_summary(client):
    client.post("/intake", json={"date": "2026-11-22", "amount_ml": 2100})
    client.post("/intake", json={"date": "2026-11-23", "amount_ml": 1000})
    weekly = client.get("/weekly?date=2026-11-23").get_json()
    assert len(weekly["days"]) == 7
    assert weekly["total_ml"] == 3100
    assert weekly["days_met_target"] == 1
    assert weekly["average_ml_per_day"] == round(3100 / 7, 1)
