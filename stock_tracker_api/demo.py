"""Demo: watchlist, buy 10 shares, and the portfolio value."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_tracker_api.app import create_app


def main():
    client = create_app().test_client()

    stocks = client.get("/stocks").get_json()
    print("Seeded prices:")
    for stock in stocks:
        print(f"  {stock['ticker']}: ${stock['price']:.2f}")

    for ticker in ("AAPL", "NVDA"):
        added = client.post("/watchlist", json={"ticker": ticker}).get_json()
        print(f"Watchlist: added {added['ticker']} at ${added['price']:.2f}")

    bought = client.post(
        "/portfolio/buy", json={"ticker": "AAPL", "shares": 10}
    ).get_json()
    print(f"Bought 10 AAPL at ${bought['price']:.2f} -> "
          f"{bought['shares']} shares worth ${bought['value']:.2f}")

    client.post("/portfolio/buy", json={"ticker": "NVDA", "shares": 5})
    portfolio = client.get("/portfolio").get_json()
    print("Portfolio:")
    for holding in portfolio["holdings"]:
        print(f"  {holding['ticker']}: {holding['shares']} shares x "
              f"${holding['price']:.2f} = ${holding['value']:.2f}")
    print(f"Total portfolio value: ${portfolio['total_value']:.2f}")

    history = client.get("/stocks/AAPL/history").get_json()
    print(f"AAPL history: {len(history['history'])} days, "
          f"latest close ${history['history'][-1]['close']:.2f}")

    oversell = client.post(
        "/portfolio/sell", json={"ticker": "AAPL", "shares": 11}
    )
    print(f"Selling 11 AAPL (only 10 held) rejected: {oversell.status_code} "
          f"{oversell.get_json()}")
    unknown = client.get("/stocks/FAKE")
    print(f"Unknown ticker rejected: {unknown.status_code} {unknown.get_json()}")


if __name__ == "__main__":
    main()
