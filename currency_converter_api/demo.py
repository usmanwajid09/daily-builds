"""Demo: list currencies, convert USD to PKR, and add a new rate."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from currency_converter_api.app import create_app


def main():
    client = create_app().test_client()

    currencies = client.get("/currencies").get_json()
    print(f"Currencies available: {len(currencies)}")
    print(f"  {', '.join(c['code'] for c in currencies)}")

    converted = client.get(
        "/convert", query_string={"from": "USD", "to": "PKR", "amount": 10}
    ).get_json()
    print(
        f"Converted: {converted['amount']} {converted['from']} = "
        f"{converted['result']} {converted['to']} "
        f"(rate {converted['rate']}, demo rates)"
    )

    added = client.post("/rates", json={"code": "CHF", "rate": 0.88})
    print(f"Added CHF at 0.88 per USD: {added.status_code} {added.get_json()}")

    bad = client.get(
        "/convert", query_string={"from": "USD", "to": "XXX", "amount": 5}
    )
    print(f"Unknown currency rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
