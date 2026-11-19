"""Tests for the budget_api Flask app, using Flask's test client."""
import pytest

from budget_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _set_budget(client, category="food", amount=20000):
    return client.post("/budgets", json={"category": category, "amount": amount})


def _expense(client, category="food", amount=1000, note=""):
    return client.post(
        "/expenses", json={"category": category, "amount": amount, "note": note}
    )


def test_list_budgets_starts_empty(client):
    assert client.get("/budgets").get_json() == []


def test_set_budget_returns_201_with_summary(client):
    resp = _set_budget(client)
    assert resp.status_code == 201
    assert resp.get_json() == {
        "category": "food",
        "budget": 20000,
        "spent": 0,
        "remaining": 20000,
        "over_budget": False,
    }


def test_set_budget_rejects_missing_category(client):
    assert client.post("/budgets", json={"amount": 100}).status_code == 400


def test_set_budget_rejects_empty_category(client):
    assert _set_budget(client, category="   ").status_code == 400


def test_set_budget_rejects_non_positive_amount(client):
    assert _set_budget(client, amount=0).status_code == 400
    assert _set_budget(client, amount=-50).status_code == 400


def test_set_budget_rejects_non_numeric_amount(client):
    assert _set_budget(client, amount="lots").status_code == 400


def test_set_budget_rejects_non_json_body(client):
    assert client.post("/budgets", data="not json").status_code == 400


def test_set_budget_overwrites_existing(client):
    _set_budget(client, amount=10000)
    resp = _set_budget(client, amount=15000)
    assert resp.get_json()["budget"] == 15000


def test_add_expense_updates_remaining(client):
    _set_budget(client)
    resp = _expense(client, amount=4500, note="Groceries")
    assert resp.status_code == 201
    assert resp.get_json()["amount"] == 4500
    summary = client.get("/budgets/food").get_json()
    assert summary["spent"] == 4500
    assert summary["remaining"] == 15500
    assert summary["over_budget"] is False


def test_over_budget_flag(client):
    _set_budget(client, amount=1000)
    _expense(client, amount=1500)
    summary = client.get("/budgets/food").get_json()
    assert summary["remaining"] == -500
    assert summary["over_budget"] is True


def test_add_expense_requires_existing_budget(client):
    assert _expense(client, category="travel").status_code == 400


def test_add_expense_rejects_bad_amount(client):
    _set_budget(client)
    assert _expense(client, amount=0).status_code == 400
    assert _expense(client, amount=-5).status_code == 400
    assert _expense(client, amount="ten").status_code == 400


def test_get_unknown_category_404(client):
    resp = client.get("/budgets/nope")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_summary_totals(client):
    _set_budget(client, "food", 20000)
    _set_budget(client, "transport", 5000)
    _expense(client, "food", 3000)
    _expense(client, "transport", 6000)
    summary = client.get("/summary").get_json()
    assert summary["total_budget"] == 25000
    assert summary["total_spent"] == 9000
    assert summary["total_remaining"] == 16000
    assert summary["over_budget"] == ["transport"]


def test_delete_budget_removes_expenses(client):
    _set_budget(client)
    _expense(client, amount=100)
    assert client.delete("/budgets/food").status_code == 200
    assert client.get("/budgets").get_json() == []
    assert client.get("/expenses").get_json() == []
    assert client.delete("/budgets/food").status_code == 404


def test_list_expenses_filter_by_category(client):
    _set_budget(client, "food", 10000)
    _set_budget(client, "transport", 5000)
    _expense(client, "food", 100)
    _expense(client, "transport", 200)
    assert len(client.get("/expenses").get_json()) == 2
    food_only = client.get("/expenses", query_string={"category": "food"}).get_json()
    assert len(food_only) == 1 and food_only[0]["category"] == "food"
