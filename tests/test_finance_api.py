"""Tests for the finance_api Flask app, using Flask's test client."""
import pytest

from finance_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _txn(client, description="Groceries", amount=8000, txn_type="expense",
         category="Food", date="2026-11-05"):
    return client.post("/transactions", json={
        "description": description, "amount": amount, "type": txn_type,
        "category": category, "date": date})


def _seed(client):
    _txn(client, "Freelance payment", 50000, "income", "Work", "2026-11-01")
    _txn(client, "Groceries", 8000, "expense", "Food", "2026-11-05")
    _txn(client, "Bus fare", 1500, "expense", "Transport", "2026-11-06")
    _txn(client, "Restaurant", 4500, "expense", "Food", "2026-11-10")


def test_list_starts_empty(client):
    assert client.get("/transactions").get_json() == []


def test_create_transaction_returns_201(client):
    resp = _txn(client)
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "description": "Groceries",
        "amount": 8000.0,
        "type": "expense",
        "category": "Food",
        "date": "2026-11-05",
    }


def test_create_amount_rounded_to_two_decimals(client):
    assert _txn(client, amount=10.999).get_json()["amount"] == 11.0


def test_create_negative_amount_400(client):
    assert _txn(client, amount=-50).status_code == 400
    assert _txn(client, amount=0).status_code == 400


def test_create_bad_type_400(client):
    assert _txn(client, txn_type="gift").status_code == 400


def test_create_bad_date_400(client):
    assert _txn(client, date="05-11-2026").status_code == 400
    assert _txn(client, date="2026-02-30").status_code == 400


def test_create_missing_description_400(client):
    resp = client.post("/transactions", json={
        "amount": 100, "type": "expense", "category": "Food",
        "date": "2026-11-05"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_create_missing_category_400(client):
    resp = client.post("/transactions", json={
        "description": "X", "amount": 100, "type": "expense",
        "date": "2026-11-05"})
    assert resp.status_code == 400


def test_get_transaction_and_404(client):
    _seed(client)
    assert client.get("/transactions/2").get_json()["description"] == "Groceries"
    resp = client.get("/transactions/99")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_filter_by_type_category_month(client):
    _seed(client)
    assert len(client.get("/transactions",
                          query_string={"type": "expense"}).get_json()) == 3
    food = client.get("/transactions",
                      query_string={"category": "food"}).get_json()
    assert [t["description"] for t in food] == ["Groceries", "Restaurant"]
    assert len(client.get("/transactions",
                          query_string={"month": "2026-11"}).get_json()) == 4
    assert client.get("/transactions",
                      query_string={"month": "2026-10"}).get_json() == []
    assert client.get("/transactions",
                      query_string={"month": "2026-13"}).status_code == 400
    assert client.get("/transactions",
                      query_string={"type": "bogus"}).status_code == 400


def test_update_transaction(client):
    _seed(client)
    resp = client.put("/transactions/2", json={"amount": 9000})
    assert resp.status_code == 200
    assert resp.get_json()["amount"] == 9000.0
    balance = client.get("/balance").get_json()
    assert balance["total_expense"] == 15000.0


def test_update_bad_amount_400(client):
    _seed(client)
    assert client.put("/transactions/2", json={"amount": -1}).status_code == 400


def test_update_unknown_404(client):
    assert client.put("/transactions/7", json={"amount": 5}).status_code == 404


def test_delete_transaction(client):
    _seed(client)
    assert client.delete("/transactions/2").get_json() == {"deleted": 2}
    assert client.get("/balance").get_json()["total_expense"] == 6000.0
    assert client.delete("/transactions/2").status_code == 404


def test_balance(client):
    _seed(client)
    balance = client.get("/balance").get_json()
    assert balance == {"total_income": 50000.0, "total_expense": 14000.0,
                       "balance": 36000.0}


def test_balance_empty(client):
    assert client.get("/balance").get_json() == {
        "total_income": 0.0, "total_expense": 0.0, "balance": 0.0}


def test_categories(client):
    _seed(client)
    cats = client.get("/categories").get_json()
    assert cats["Food"] == {"income": 0.0, "expense": 12500.0, "count": 2}
    assert cats["Work"] == {"income": 50000.0, "expense": 0.0, "count": 1}
    assert cats["Transport"]["expense"] == 1500.0


def test_summary_month(client):
    _seed(client)
    _txn(client, "Old salary", 1000, "income", "Work", "2026-10-01")
    summary = client.get("/summary",
                         query_string={"month": "2026-11"}).get_json()
    assert summary["transaction_count"] == 4
    assert summary["total_income"] == 50000.0
    assert summary["total_expense"] == 14000.0
    assert summary["balance"] == 36000.0
    assert summary["by_category"]["Food"]["expense"] == 12500.0


def test_summary_all_time_and_bad_month(client):
    _seed(client)
    summary = client.get("/summary").get_json()
    assert summary["month"] is None
    assert summary["transaction_count"] == 4
    assert client.get("/summary",
                      query_string={"month": "Nov"}).status_code == 400


def test_create_budget(client):
    resp = client.post("/budgets", json={"category": "Food", "limit": 10000})
    assert resp.status_code == 201
    assert resp.get_json() == {"category": "Food", "limit": 10000.0}


def test_create_budget_duplicate_400(client):
    client.post("/budgets", json={"category": "Food", "limit": 10000})
    assert client.post("/budgets",
                       json={"category": "Food", "limit": 5000}).status_code == 400


def test_create_budget_bad_limit_400(client):
    assert client.post("/budgets",
                       json={"category": "Food", "limit": 0}).status_code == 400
    assert client.post("/budgets",
                       json={"category": "Food", "limit": -5}).status_code == 400


def test_list_budgets_sorted(client):
    client.post("/budgets", json={"category": "Transport", "limit": 2000})
    client.post("/budgets", json={"category": "Food", "limit": 10000})
    budgets = client.get("/budgets").get_json()
    assert [b["category"] for b in budgets] == ["Food", "Transport"]


def test_update_and_delete_budget(client):
    client.post("/budgets", json={"category": "Food", "limit": 10000})
    resp = client.put("/budgets/Food", json={"limit": 12000})
    assert resp.get_json()["limit"] == 12000.0
    assert client.put("/budgets/Nope", json={"limit": 5}).status_code == 404
    assert client.delete("/budgets/Food").get_json() == {"deleted": "Food"}
    assert client.get("/budgets").get_json() == []
    assert client.delete("/budgets/Food").status_code == 404


def test_budget_alerts_statuses(client):
    _seed(client)
    client.post("/budgets", json={"category": "Food", "limit": 10000})
    client.post("/budgets", json={"category": "Transport", "limit": 2000})
    client.post("/budgets", json={"category": "Work", "limit": 1000})
    alerts = {a["category"]: a for a in client.get("/budgets/alerts").get_json()}
    assert alerts["Food"]["status"] == "over"
    assert alerts["Food"]["spent"] == 12500.0
    assert alerts["Food"]["pct_used"] == 125.0
    assert alerts["Food"]["remaining"] == -2500.0
    assert alerts["Transport"]["status"] == "ok"
    assert alerts["Transport"]["pct_used"] == 75.0
    assert alerts["Transport"]["remaining"] == 500.0
    assert alerts["Work"]["status"] == "ok"
    assert alerts["Work"]["spent"] == 0.0


def test_budget_alerts_warning_boundary(client):
    _txn(client, "Snack", 800, "expense", "Food", "2026-11-05")
    client.post("/budgets", json={"category": "Food", "limit": 1000})
    alert = client.get("/budgets/alerts").get_json()[0]
    assert alert["pct_used"] == 80.0
    assert alert["status"] == "warning"
