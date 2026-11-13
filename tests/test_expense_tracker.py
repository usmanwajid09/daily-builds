"""Tests for the expense_tracker Flask app, using Flask's test client."""
import pytest

from expense_tracker.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _add(client, amount=10.0, category="food", date="2026-11-01", desc=""):
    resp = client.post(
        "/expenses",
        json={
            "amount": amount,
            "category": category,
            "date": date,
            "description": desc,
        },
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()


def test_add_expense_returns_201(client):
    expense = _add(client, 12.5, "food", "2026-11-01", "Lunch")
    assert expense == {
        "id": 1,
        "amount": 12.5,
        "category": "food",
        "date": "2026-11-01",
        "description": "Lunch",
    }


def test_add_rejects_bad_amounts(client):
    for bad in (0, -5, "10", None, True):
        resp = client.post(
            "/expenses",
            json={"amount": bad, "category": "food", "date": "2026-11-01"},
        )
        assert resp.status_code == 400, bad
        assert "amount" in resp.get_json()["error"]


def test_add_rejects_missing_category_and_bad_dates(client):
    resp = client.post(
        "/expenses", json={"amount": 5, "category": " ", "date": "2026-11-01"}
    )
    assert resp.status_code == 400
    for bad_date in ("01-11-2026", "2026-13-01", "2026-02-30", "not-a-date"):
        resp = client.post(
            "/expenses",
            json={"amount": 5, "category": "food", "date": bad_date},
        )
        assert resp.status_code == 400, bad_date
        assert "date" in resp.get_json()["error"]


def test_add_rejects_non_json_body(client):
    resp = client.post("/expenses", data="x", content_type="text/plain")
    assert resp.status_code == 400


def test_list_and_filter_by_category(client):
    _add(client, 10, "food", "2026-11-01")
    _add(client, 20, "transport", "2026-11-02")
    assert len(client.get("/expenses").get_json()) == 2
    food = client.get("/expenses?category=food").get_json()
    assert [e["category"] for e in food] == ["food"]


def test_get_and_delete_round_trip(client):
    expense = _add(client)
    assert client.get(f"/expenses/{expense['id']}").get_json() == expense
    assert client.delete(f"/expenses/{expense['id']}").get_json() == {
        "deleted": expense["id"]
    }
    assert client.get(f"/expenses/{expense['id']}").status_code == 404


def test_unknown_expense_is_404(client):
    assert client.get("/expenses/999").status_code == 404
    assert client.delete("/expenses/999").status_code == 404


def test_totals_per_category(client):
    _add(client, 12.50, "food", "2026-11-01")
    _add(client, 45.00, "transport", "2026-11-02")
    _add(client, 8.75, "food", "2026-11-03")
    totals = client.get("/totals").get_json()
    assert totals["by_category"] == {"food": 21.25, "transport": 45.0}
    assert totals["total"] == 66.25
    assert totals["count"] == 3


def test_monthly_total_and_bad_month(client):
    _add(client, 10, "food", "2026-11-01")
    _add(client, 20, "food", "2026-12-01")
    assert client.get("/totals/2026-11").get_json() == {
        "month": "2026-11",
        "total": 10.0,
        "count": 1,
    }
    assert client.get("/totals/2026-10").get_json()["total"] == 0.0
    resp = client.get("/totals/november")
    assert resp.status_code == 400
