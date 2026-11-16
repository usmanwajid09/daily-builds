"""Tests for the invoice_api Flask app, using Flask's test client."""
import pytest

from invoice_api.app import create_app

INVOICE = {
    "customer": "Acme Ltd",
    "items": [
        {"description": "Website design", "quantity": 1, "unit_price": 500.00},
        {"description": "Hosting (months)", "quantity": 3, "unit_price": 20.00},
    ],
    "tax_rate": 0.10,
}


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_list_starts_empty(client):
    assert client.get("/invoices").get_json() == []


def test_create_computes_totals(client):
    resp = client.post("/invoices", json=INVOICE)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["id"] == 1
    assert body["subtotal"] == 560.00
    assert body["tax"] == 56.00
    assert body["total"] == 616.00
    assert body["status"] == "unpaid"
    assert body["items"][1]["line_total"] == 60.00


def test_create_uses_default_tax_rate(client):
    payload = {k: v for k, v in INVOICE.items() if k != "tax_rate"}
    body = client.post("/invoices", json=payload).get_json()
    assert body["tax_rate"] == 0.10
    assert body["total"] == 616.00


def test_create_rounds_to_two_decimals(client):
    payload = {
        "customer": "Rounding Co",
        "items": [{"description": "Odd price", "quantity": 3, "unit_price": 3.33}],
        "tax_rate": 0.07,
    }
    body = client.post("/invoices", json=payload).get_json()
    assert body["subtotal"] == 9.99
    assert body["tax"] == 0.70
    assert body["total"] == 10.69


def test_create_rejects_missing_customer(client):
    payload = dict(INVOICE)
    del payload["customer"]
    resp = client.post("/invoices", json=payload)
    assert resp.status_code == 400
    assert "customer" in resp.get_json()["error"]


def test_create_rejects_bad_items(client):
    for bad_items in ([], "not a list", [{"description": "x"}],
                      [{"description": "x", "quantity": 0, "unit_price": 5}],
                      [{"description": "x", "quantity": 1, "unit_price": -5}],
                      [{"description": "  ", "quantity": 1, "unit_price": 5}]):
        resp = client.post("/invoices", json={"customer": "C", "items": bad_items})
        assert resp.status_code == 400


def test_create_rejects_bad_tax_rate(client):
    for bad in (-0.1, 1.5, "10%", True):
        resp = client.post("/invoices", json=dict(INVOICE, tax_rate=bad))
        assert resp.status_code == 400


def test_create_rejects_non_json_body(client):
    resp = client.post("/invoices", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_pay_marks_paid_and_second_pay_is_400(client):
    invoice_id = client.post("/invoices", json=INVOICE).get_json()["id"]
    paid = client.post(f"/invoices/{invoice_id}/pay").get_json()
    assert paid["status"] == "paid"
    assert paid["total"] == 616.00
    resp = client.post(f"/invoices/{invoice_id}/pay")
    assert resp.status_code == 400
    assert "already paid" in resp.get_json()["error"]


def test_list_filter_by_status(client):
    first = client.post("/invoices", json=INVOICE).get_json()["id"]
    client.post("/invoices", json=dict(INVOICE, customer="Beta Co"))
    client.post(f"/invoices/{first}/pay")
    assert len(client.get("/invoices").get_json()) == 2
    assert [i["id"] for i in client.get("/invoices?status=paid").get_json()] == [first]
    assert len(client.get("/invoices?status=unpaid").get_json()) == 1
    assert client.get("/invoices?status=draft").status_code == 400


def test_get_and_delete_round_trip(client):
    invoice_id = client.post("/invoices", json=INVOICE).get_json()["id"]
    assert client.get(f"/invoices/{invoice_id}").get_json()["customer"] == "Acme Ltd"
    assert client.delete(f"/invoices/{invoice_id}").get_json() == {"deleted": invoice_id}
    assert client.get(f"/invoices/{invoice_id}").status_code == 404


def test_unknown_invoice_is_404_json(client):
    for method, path in (
        ("get", "/invoices/999"),
        ("post", "/invoices/999/pay"),
        ("delete", "/invoices/999"),
    ):
        resp = getattr(client, method)(path, json={})
        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"]


def test_unknown_route_is_404_json(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert "error" in resp.get_json()
