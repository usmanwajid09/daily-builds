"""Tests for the gym membership API."""
import pytest

from gym_membership_api.app import create_app


@pytest.fixture()
def client():
    return create_app().test_client()


def _plan(client, name="Monthly", days=30, price=5000):
    resp = client.post("/plans", json={"name": name, "duration_days": days, "price": price})
    assert resp.status_code == 201
    return resp.get_json()


def _member(client, plan_id, name="Usman", start="2026-11-25"):
    resp = client.post("/members", json={"name": name, "plan_id": plan_id, "start_date": start})
    assert resp.status_code == 201
    return resp.get_json()


def test_create_plan(client):
    plan = _plan(client)
    assert plan["name"] == "Monthly" and plan["duration_days"] == 30 and plan["price"] == 5000


def test_create_plan_bad_duration(client):
    resp = client.post("/plans", json={"name": "X", "duration_days": 0, "price": 100})
    assert resp.status_code == 400


def test_member_expiry_computed(client):
    plan = _plan(client, days=30)
    member = _member(client, plan["id"])
    assert member["start_date"] == "2026-11-25"
    assert member["expiry_date"] == "2026-12-25"
    assert member["expired"] is False
    assert member["days_remaining"] == 30


def test_member_unknown_plan_404(client):
    resp = client.post("/members", json={"name": "Usman", "plan_id": 99})
    assert resp.status_code == 404


def test_member_bad_start_date(client):
    plan = _plan(client)
    resp = client.post("/members", json={"name": "Usman", "plan_id": plan["id"], "start_date": "25/11/2026"})
    assert resp.status_code == 400


def test_expired_member_flagged(client):
    plan = _plan(client, days=30)
    old = _member(client, plan["id"], name="Ali", start="2026-09-01")
    assert old["expired"] is True
    assert old["days_remaining"] == 0
    assert old["expiry_date"] == "2026-10-01"


def test_expired_list(client):
    plan = _plan(client, days=30)
    _member(client, plan["id"], name="Usman")
    _member(client, plan["id"], name="Ali", start="2026-09-01")
    expired = client.get("/members/expired").get_json()
    assert [m["name"] for m in expired] == ["Ali"]


def test_check_in_success(client):
    plan = _plan(client)
    member = _member(client, plan["id"])
    resp = client.post(f"/members/{member['id']}/check-in", json={"date": "2026-11-25"})
    assert resp.status_code == 201
    assert resp.get_json()["date"] == "2026-11-25"
    fetched = client.get(f"/members/{member['id']}").get_json()
    assert fetched["checkin_count"] == 1


def test_duplicate_check_in_409(client):
    plan = _plan(client)
    member = _member(client, plan["id"])
    assert client.post(f"/members/{member['id']}/check-in", json={"date": "2026-11-25"}).status_code == 201
    assert client.post(f"/members/{member['id']}/check-in", json={"date": "2026-11-25"}).status_code == 409


def test_expired_check_in_409(client):
    plan = _plan(client, days=30)
    old = _member(client, plan["id"], name="Ali", start="2026-09-01")
    resp = client.post(f"/members/{old['id']}/check-in", json={"date": "2026-11-25"})
    assert resp.status_code == 409


def test_check_in_unknown_member_404(client):
    assert client.post("/members/99/check-in", json={}).status_code == 404


def test_list_checkins(client):
    plan = _plan(client)
    member = _member(client, plan["id"])
    client.post(f"/members/{member['id']}/check-in", json={"date": "2026-11-25"})
    client.post(f"/members/{member['id']}/check-in", json={"date": "2026-11-26"})
    result = client.get(f"/members/{member['id']}/check-ins").get_json()
    assert result["count"] == 2


def test_delete_member(client):
    plan = _plan(client)
    member = _member(client, plan["id"])
    client.post(f"/members/{member['id']}/check-in", json={"date": "2026-11-25"})
    assert client.delete(f"/members/{member['id']}").status_code == 200
    assert client.get(f"/members/{member['id']}").status_code == 404
    assert client.get(f"/members/{member['id']}/check-ins").status_code == 404


def test_get_plan_404(client):
    assert client.get("/plans/99").status_code == 404
