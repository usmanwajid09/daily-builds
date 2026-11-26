"""Tests for the stock_tracker_api Flask app, using Flask's test client."""
import pytest

from stock_tracker_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_list_stocks_seeded(client):
    stocks = client.get("/stocks").get_json()
    assert {s["ticker"] for s in stocks} == {"AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"}
    prices = {s["ticker"]: s["price"] for s in stocks}
    assert prices["AAPL"] == 175.50
    assert prices["MSFT"] == 420.25


def test_get_stock_and_404(client):
    resp = client.get("/stocks/AAPL")
    assert resp.status_code == 200
    assert resp.get_json() == {"ticker": "AAPL", "price": 175.50}
    assert client.get("/stocks/FAKE").status_code == 404


def test_get_stock_ticker_is_case_insensitive(client):
    assert client.get("/stocks/aapl").get_json()["price"] == 175.50


def test_price_history(client):
    resp = client.get("/stocks/AAPL/history")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ticker"] == "AAPL"
    assert len(body["history"]) == 7
    assert body["history"][0]["date"] == "2026-11-20"
    assert body["history"][-1] == {"date": "2026-11-26", "close": 175.50}
    assert client.get("/stocks/FAKE/history").status_code == 404


def test_watchlist_add_and_list(client):
    assert client.get("/watchlist").get_json() == []
    resp = client.post("/watchlist", json={"ticker": "aapl"})
    assert resp.status_code == 201
    assert resp.get_json() == {"ticker": "AAPL", "price": 175.50}
    assert client.get("/watchlist").get_json() == [
        {"ticker": "AAPL", "price": 175.50}
    ]


def test_watchlist_duplicate_returns_409(client):
    client.post("/watchlist", json={"ticker": "AAPL"})
    dup = client.post("/watchlist", json={"ticker": "aapl"})
    assert dup.status_code == 409
    assert "already" in dup.get_json()["error"]


def test_watchlist_unknown_ticker_is_404(client):
    assert client.post("/watchlist", json={"ticker": "FAKE"}).status_code == 404
    assert client.post("/watchlist", json={}).status_code == 400


def test_watchlist_remove(client):
    client.post("/watchlist", json={"ticker": "AAPL"})
    resp = client.delete("/watchlist/AAPL")
    assert resp.get_json() == {"removed": "AAPL"}
    assert client.get("/watchlist").get_json() == []
    assert client.delete("/watchlist/AAPL").status_code == 404


def test_portfolio_starts_empty(client):
    assert client.get("/portfolio").get_json() == {"holdings": [], "total_value": 0.0}


def test_buy_shares_and_portfolio_value(client):
    resp = client.post("/portfolio/buy", json={"ticker": "AAPL", "shares": 10})
    assert resp.status_code == 200
    assert resp.get_json() == {
        "ticker": "AAPL",
        "shares": 10,
        "price": 175.50,
        "value": 1755.00,
    }
    client.post("/portfolio/buy", json={"ticker": "NVDA", "shares": 5})
    portfolio = client.get("/portfolio").get_json()
    assert portfolio["total_value"] == 2433.00
    assert [h["ticker"] for h in portfolio["holdings"]] == ["AAPL", "NVDA"]
    assert portfolio["holdings"][1]["value"] == 678.00


def test_buy_accumulates_shares(client):
    client.post("/portfolio/buy", json={"ticker": "AAPL", "shares": 10})
    resp = client.post("/portfolio/buy", json={"ticker": "AAPL", "shares": 4})
    assert resp.get_json()["shares"] == 14
    assert resp.get_json()["value"] == 2457.00


def test_buy_validation_errors(client):
    assert client.post("/portfolio/buy", json={"ticker": "FAKE", "shares": 1}).status_code == 404
    assert client.post("/portfolio/buy", json={"ticker": "AAPL", "shares": 0}).status_code == 400
    assert client.post("/portfolio/buy", json={"ticker": "AAPL", "shares": -3}).status_code == 400
    assert client.post("/portfolio/buy", json={"ticker": "AAPL", "shares": 1.5}).status_code == 400
    assert client.post("/portfolio/buy", json={"ticker": "AAPL"}).status_code == 400


def test_sell_shares(client):
    client.post("/portfolio/buy", json={"ticker": "AAPL", "shares": 10})
    resp = client.post("/portfolio/sell", json={"ticker": "AAPL", "shares": 4})
    assert resp.status_code == 200
    assert resp.get_json()["shares"] == 6
    assert resp.get_json()["value"] == 1053.00


def test_sell_all_removes_holding(client):
    client.post("/portfolio/buy", json={"ticker": "AAPL", "shares": 10})
    client.post("/portfolio/sell", json={"ticker": "AAPL", "shares": 10})
    assert client.get("/portfolio").get_json() == {"holdings": [], "total_value": 0.0}


def test_sell_more_than_held_returns_400(client):
    client.post("/portfolio/buy", json={"ticker": "AAPL", "shares": 10})
    resp = client.post("/portfolio/sell", json={"ticker": "AAPL", "shares": 11})
    assert resp.status_code == 400
    assert "only 10 held" in resp.get_json()["error"]
    # Selling a ticker never bought is also a 400.
    assert client.post("/portfolio/sell", json={"ticker": "NVDA", "shares": 1}).status_code == 400


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
