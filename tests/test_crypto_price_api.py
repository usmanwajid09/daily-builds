"""Tests for the crypto_price_api Flask app, using Flask's test client."""
import pytest

from crypto_price_api.app import COINS, create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_list_coins(client):
    coins = client.get("/coins").get_json()
    assert len(coins) == 8
    symbols = [c["symbol"] for c in coins]
    assert symbols == sorted(symbols)
    assert set(symbols) == {"BTC", "ETH", "SOL", "DOGE", "ADA", "XRP", "LTC", "LINK"}


def test_list_coin_fields_match_seed(client):
    coins = {c["symbol"]: c for c in client.get("/coins").get_json()}
    assert coins["BTC"]["price_usd"] == 97432.10
    assert coins["BTC"]["name"] == "Bitcoin"
    assert coins["DOGE"]["price_usd"] == 0.4123
    assert coins["BTC"]["market_cap_usd"] == round(97432.10 * 19_800_000, 2)


def test_change_24h_matches_history(client):
    btc = client.get("/coins/BTC").get_json()
    history = client.get("/coins/BTC/history",
                         query_string={"days": 2}).get_json()["prices"]
    expected = round((history[1]["price_usd"] - history[0]["price_usd"])
                     / history[0]["price_usd"] * 100, 2)
    assert btc["change_24h_pct"] == expected


def test_get_coin_detail_and_7d_change(client):
    eth = client.get("/coins/eth").get_json()
    assert eth["symbol"] == "ETH"
    assert eth["price_usd"] == 3412.55
    history = COINS["ETH"]["history"]
    expected_7d = round((history[-1] - history[-8]) / history[-8] * 100, 2)
    assert eth["change_7d_pct"] == expected_7d


def test_get_unknown_coin_404(client):
    resp = client.get("/coins/FAKE")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_history_default_7_days(client):
    result = client.get("/coins/BTC/history").get_json()
    assert result["days"] == 7
    assert len(result["prices"]) == 7
    assert result["prices"][-1]["price_usd"] == 97432.10


def test_history_custom_days(client):
    result = client.get("/coins/SOL/history",
                        query_string={"days": 30}).get_json()
    assert len(result["prices"]) == 30
    assert result["prices"][-1]["price_usd"] == 238.40


def test_history_dates_are_consecutive(client):
    from datetime import date, timedelta
    result = client.get("/coins/BTC/history",
                        query_string={"days": 3}).get_json()
    dates = [p["date"] for p in result["prices"]]
    today = date.today()
    assert dates == [(today - timedelta(days=d)).isoformat() for d in (2, 1, 0)]


def test_history_bad_days_400(client):
    assert client.get("/coins/BTC/history",
                      query_string={"days": 0}).status_code == 400
    assert client.get("/coins/BTC/history",
                      query_string={"days": 31}).status_code == 400
    assert client.get("/coins/BTC/history",
                      query_string={"days": "abc"}).status_code == 400


def test_history_unknown_coin_404(client):
    assert client.get("/coins/NOPE/history").status_code == 404


def test_change_endpoint(client):
    result = client.get("/coins/ETH/change",
                        query_string={"days": 7}).get_json()
    history = COINS["ETH"]["history"]
    assert result["start_price_usd"] == history[-8]
    assert result["end_price_usd"] == history[-1]
    expected = round((history[-1] - history[-8]) / history[-8] * 100, 2)
    assert result["change_pct"] == expected


def test_change_bad_days_400(client):
    assert client.get("/coins/ETH/change",
                      query_string={"days": 99}).status_code == 400


def test_convert_btc_to_eth(client):
    result = client.get("/convert", query_string={
        "from": "BTC", "to": "ETH", "amount": 2}).get_json()
    expected_rate = round(97432.10 / 3412.55, 6)
    assert result["rate"] == expected_rate
    assert result["converted"] == round(2 * expected_rate, 6)
    assert result["amount"] == 2.0


def test_convert_defaults_to_one_and_same_coin(client):
    result = client.get("/convert", query_string={"from": "SOL", "to": "SOL"}).get_json()
    assert result["amount"] == 1.0
    assert result["rate"] == 1.0
    assert result["converted"] == 1.0


def test_convert_unknown_coin_400(client):
    assert client.get("/convert", query_string={
        "from": "BTC", "to": "FAKE"}).status_code == 400
    assert client.get("/convert", query_string={
        "from": "FAKE", "to": "ETH"}).status_code == 400


def test_convert_bad_amount_400(client):
    assert client.get("/convert", query_string={
        "from": "BTC", "to": "ETH", "amount": 0}).status_code == 400
    assert client.get("/convert", query_string={
        "from": "BTC", "to": "ETH", "amount": -1}).status_code == 400
    assert client.get("/convert", query_string={
        "from": "BTC", "to": "ETH", "amount": "lots"}).status_code == 400


def test_top_by_market_cap(client):
    top = client.get("/top", query_string={"limit": 3}).get_json()
    assert [c["symbol"] for c in top] == ["BTC", "ETH", "XRP"]
    caps = [c["market_cap_usd"] for c in top]
    assert caps == sorted(caps, reverse=True)


def test_top_default_limit_5(client):
    assert len(client.get("/top").get_json()) == 5


def test_top_bad_limit_400(client):
    assert client.get("/top", query_string={"limit": 0}).status_code == 400
    assert client.get("/top", query_string={"limit": 9}).status_code == 400
    assert client.get("/top", query_string={"limit": "x"}).status_code == 400


def test_prices_are_deterministic(client):
    first = client.get("/coins").get_json()
    second = client.get("/coins").get_json()
    assert first == second
    assert client.get("/coins/BTC/history").get_json() == \
        client.get("/coins/BTC/history").get_json()
