"""Demo: list coins, inspect BTC history/change, convert BTC to ETH, top coins."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_price_api.app import create_app


def main():
    client = create_app().test_client()

    coins = client.get("/coins").get_json()
    print(f"Coins ({len(coins)}): "
          + ", ".join(f"{c['symbol']} ${c['price_usd']}" for c in coins))

    btc = client.get("/coins/btc").get_json()
    print(f"BTC: ${btc['price_usd']}, 24h {btc['change_24h_pct']}%, "
          f"7d {btc['change_7d_pct']}%, market cap ${btc['market_cap_usd']:,.0f}")

    history = client.get("/coins/BTC/history", query_string={"days": 7}).get_json()
    print(f"BTC 7-day history: "
          + " -> ".join(f"${p['price_usd']}" for p in history["prices"]))

    change = client.get("/coins/ETH/change", query_string={"days": 7}).get_json()
    print(f"ETH 7-day change: ${change['start_price_usd']} -> "
          f"${change['end_price_usd']} ({change['change_pct']}%)")

    converted = client.get(
        "/convert", query_string={"from": "BTC", "to": "ETH", "amount": 2}
    ).get_json()
    print(f"Convert: 2 BTC = {converted['converted']} ETH "
          f"(rate {converted['rate']} ETH/BTC)")

    top = client.get("/top", query_string={"limit": 3}).get_json()
    print(f"Top 3 by market cap: {[c['symbol'] for c in top]}")

    bad = client.get("/coins/FAKE")
    print(f"Unknown coin rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
