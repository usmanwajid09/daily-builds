"""Tests for the currency_converter_api Flask app."""
import pytest

from currency_converter_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_list_currencies_includes_defaults(client):
    currencies = client.get("/currencies").get_json()
    codes = [c["code"] for c in currencies]
    assert codes == sorted(codes)
    assert "USD" in codes and "PKR" in codes
    pkr = next(c for c in currencies if c["code"] == "PKR")
    assert pkr["rate_per_usd"] == 278.0


def test_convert_usd_to_pkr(client):
    resp = client.get(
        "/convert", query_string={"from": "USD", "to": "PKR", "amount": 10}
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data == {
        "from": "USD",
        "to": "PKR",
        "amount": 10.0,
        "rate": 278.0,
        "result": 2780.0,
        "demo_rates": True,
    }


def test_convert_cross_rate_and_lowercase_codes(client):
    data = client.get(
        "/convert", query_string={"from": "eur", "to": "gbp", "amount": 100}
    ).get_json()
    assert data["from"] == "EUR" and data["to"] == "GBP"
    assert data["rate"] == pytest.approx(0.79 / 0.92)
    assert data["result"] == pytest.approx(100 * 0.79 / 0.92)


def test_convert_same_currency(client):
    data = client.get(
        "/convert", query_string={"from": "USD", "to": "USD", "amount": 5}
    ).get_json()
    assert data["rate"] == 1.0 and data["result"] == 5.0


def test_convert_rejects_bad_input(client):
    base = {"from": "USD", "to": "PKR", "amount": 10}
    assert client.get("/convert", query_string={**base, "to": "XXX"}).status_code == 400
    assert client.get("/convert", query_string={**base, "from": "XX"}).status_code == 400
    assert client.get("/convert", query_string={**base, "amount": "abc"}).status_code == 400
    assert client.get("/convert", query_string={**base, "amount": "0"}).status_code == 400
    assert client.get("/convert", query_string={**base, "amount": "-5"}).status_code == 400
    assert client.get("/convert", query_string={"from": "USD", "to": "PKR"}).status_code == 400
    assert client.get("/convert", query_string={"to": "PKR", "amount": 1}).status_code == 400


def test_get_rate_and_404(client):
    assert client.get("/rates/EUR").get_json() == {"code": "EUR", "rate_per_usd": 0.92}
    assert client.get("/rates/eur").get_json()["code"] == "EUR"
    assert client.get("/rates/XXX").status_code == 404
    assert client.get("/rates/US").status_code == 400


def test_add_rate_returns_201_then_usable(client):
    resp = client.post("/rates", json={"code": "CHF", "rate": 0.88})
    assert resp.status_code == 201
    assert resp.get_json() == {"code": "CHF", "rate_per_usd": 0.88}
    converted = client.get(
        "/convert", query_string={"from": "USD", "to": "CHF", "amount": 10}
    ).get_json()
    assert converted["result"] == pytest.approx(8.8)


def test_update_rate_returns_200(client):
    resp = client.post("/rates", json={"code": "eur", "rate": 0.95})
    assert resp.status_code == 200
    assert client.get("/rates/EUR").get_json()["rate_per_usd"] == 0.95


def test_add_rate_rejects_bad_input(client):
    assert client.post("/rates", json={"code": "EU", "rate": 0.9}).status_code == 400
    assert client.post("/rates", json={"code": "EUR1", "rate": 0.9}).status_code == 400
    assert client.post("/rates", json={"code": "EUR", "rate": 0}).status_code == 400
    assert client.post("/rates", json={"code": "EUR", "rate": -1}).status_code == 400
    assert client.post("/rates", json={"code": "EUR", "rate": "high"}).status_code == 400
    assert client.post("/rates", json={"code": "EUR"}).status_code == 400
    assert client.post("/rates", json={"rate": 0.9}).status_code == 400


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
