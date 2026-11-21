"""Tests for the bill_splitter_api Flask app, using Flask's test client."""
import pytest

from bill_splitter_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _bill(client, title="Dinner", total=100, payer="Usman",
          participants=None, **extra):
    payload = {
        "title": title,
        "total": total,
        "payer": payer,
        "participants": participants or ["Usman", "Ali", "Sara", "Bilal"],
    }
    payload.update(extra)
    return client.post("/bills", json=payload)


def test_list_starts_empty(client):
    assert client.get("/bills").get_json() == []


def test_create_bill_returns_201_equal_split(client):
    resp = _bill(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["shares"] == {"Usman": 25.0, "Ali": 25.0,
                              "Sara": 25.0, "Bilal": 25.0}
    assert body["outstanding"] == 75.0
    assert body["settled"] is False
    assert body["payments"] == []


def test_equal_split_rounding_remainder(client):
    body = _bill(client, total=100,
                 participants=["Usman", "Ali", "Sara"]).get_json()
    assert body["shares"] == {"Usman": 33.33, "Ali": 33.33,
                              "Sara": 33.34}
    assert sum(body["shares"].values()) == 100.0


def test_create_custom_shares(client):
    resp = _bill(client, title="Groceries", total=90, payer="Ali",
                 participants=["Ali", "Sara"],
                 shares={"Ali": 60, "Sara": 30})
    assert resp.status_code == 201
    assert resp.get_json()["shares"] == {"Ali": 60.0, "Sara": 30.0}


def test_create_missing_title_400(client):
    resp = client.post("/bills", json={"total": 50, "payer": "Usman",
                                       "participants": ["Usman", "Ali"]})
    assert resp.status_code == 400


def test_create_bad_total_400(client):
    assert _bill(client, total=0).status_code == 400
    assert _bill(client, total=-10).status_code == 400
    assert _bill(client, total="free").status_code == 400


def test_create_payer_not_participant_400(client):
    assert _bill(client, payer="Zara").status_code == 400


def test_create_too_few_participants_400(client):
    assert _bill(client, participants=["Usman"]).status_code == 400
    resp = client.post("/bills", json={"title": "Solo", "total": 50,
                                       "payer": "Usman", "participants": []})
    assert resp.status_code == 400


def test_create_duplicate_participants_400(client):
    assert _bill(client,
                 participants=["Usman", "Ali", "Ali"]).status_code == 400


def test_create_shares_not_summing_400(client):
    assert _bill(client, participants=["Usman", "Ali"],
                 shares={"Usman": 40, "Ali": 40}).status_code == 400


def test_create_shares_missing_participant_400(client):
    assert _bill(client, participants=["Usman", "Ali", "Sara"],
                 shares={"Usman": 50, "Ali": 50}).status_code == 400


def test_get_bill(client):
    _bill(client)
    resp = client.get("/bills/1")
    assert resp.status_code == 200
    assert resp.get_json()["title"] == "Dinner"


def test_get_missing_404(client):
    assert client.get("/bills/99").status_code == 404


def test_payment_records_and_updates_balance(client):
    _bill(client)
    resp = client.post("/bills/1/payments",
                       json={"person": "Ali", "amount": 25})
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["payments"] == [{"id": 1, "person": "Ali", "amount": 25.0}]
    assert body["outstanding"] == 50.0
    ali = next(r for r in body["balances"] if r["person"] == "Ali")
    assert ali == {"person": "Ali", "share": 25.0, "paid": 25.0,
                   "balance": 0.0, "is_payer": False}


def test_partial_payment_leaves_balance(client):
    _bill(client)
    client.post("/bills/1/payments", json={"person": "Sara", "amount": 10})
    rows = client.get("/bills/1/balances").get_json()
    sara = next(r for r in rows if r["person"] == "Sara")
    assert sara["balance"] == 15.0
    usman = next(r for r in rows if r["person"] == "Usman")
    assert usman["balance"] == 0.0
    assert usman["is_payer"] is True


def test_payment_overpay_400(client):
    _bill(client)
    assert client.post("/bills/1/payments",
                       json={"person": "Ali", "amount": 26}).status_code == 400
    client.post("/bills/1/payments", json={"person": "Ali", "amount": 20})
    assert client.post("/bills/1/payments",
                       json={"person": "Ali", "amount": 6}).status_code == 400


def test_payment_by_non_participant_400(client):
    _bill(client)
    assert client.post("/bills/1/payments",
                       json={"person": "Zara", "amount": 5}).status_code == 400


def test_payment_by_payer_400(client):
    _bill(client)
    assert client.post("/bills/1/payments",
                       json={"person": "Usman", "amount": 5}).status_code == 400


def test_payment_bad_amount_400(client):
    _bill(client)
    assert client.post("/bills/1/payments",
                       json={"person": "Ali", "amount": 0}).status_code == 400
    assert client.post("/bills/1/payments",
                       json={"person": "Ali", "amount": -5}).status_code == 400


def test_payment_missing_bill_404(client):
    assert client.post("/bills/99/payments",
                       json={"person": "Ali", "amount": 5}).status_code == 404


def test_settlements_plan(client):
    _bill(client)
    client.post("/bills/1/payments", json={"person": "Ali", "amount": 25})
    client.post("/bills/1/payments", json={"person": "Sara", "amount": 10})
    plan = client.get("/bills/1/settlements").get_json()
    assert plan == [
        {"from": "Sara", "to": "Usman", "amount": 15.0},
        {"from": "Bilal", "to": "Usman", "amount": 25.0},
    ]


def test_settlements_empty_when_settled(client):
    _bill(client, participants=["Usman", "Ali"])
    client.post("/bills/1/payments", json={"person": "Ali", "amount": 50})
    assert client.get("/bills/1/settlements").get_json() == []
    assert client.get("/bills/1").get_json()["settled"] is True


def test_update_total_recomputes_shares(client):
    _bill(client)
    resp = client.put("/bills/1", json={"total": 200})
    assert resp.status_code == 200
    assert resp.get_json()["shares"] == {"Usman": 50.0, "Ali": 50.0,
                                         "Sara": 50.0, "Bilal": 50.0}


def test_update_remove_paying_participant_400(client):
    _bill(client)
    client.post("/bills/1/payments", json={"person": "Ali", "amount": 10})
    resp = client.put("/bills/1",
                      json={"participants": ["Usman", "Sara", "Bilal"]})
    assert resp.status_code == 400


def test_update_nothing_400(client):
    _bill(client)
    assert client.put("/bills/1", json={}).status_code == 400


def test_update_missing_404(client):
    assert client.put("/bills/99", json={"title": "X"}).status_code == 404


def test_delete_bill(client):
    _bill(client)
    assert client.delete("/bills/1").get_json() == {"deleted": 1}
    assert client.get("/bills/1").status_code == 404


def test_summary(client):
    _bill(client, title="Dinner", total=100)
    _bill(client, title="Taxi", total=40, payer="Ali",
          participants=["Ali", "Usman"])
    client.post("/bills/2/payments", json={"person": "Usman", "amount": 20})
    summary = client.get("/summary").get_json()
    assert summary["total_bills"] == 2
    assert summary["total_amount"] == 140.0
    assert summary["total_outstanding"] == 75.0
    assert summary["settled_bills"] == 1


def test_unknown_route_404(client):
    assert client.get("/nope").status_code == 404
