"""Demo: create invoices, check auto-computed totals, mark paid, filter by status."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invoice_api.app import create_app


def main():
    client = create_app().test_client()

    invoice = client.post(
        "/invoices",
        json={
            "customer": "Acme Ltd",
            "items": [
                {"description": "Website design", "quantity": 1, "unit_price": 500.00},
                {"description": "Hosting (months)", "quantity": 3, "unit_price": 20.00},
            ],
            "tax_rate": 0.10,
        },
    ).get_json()
    print(f"Created invoice: {invoice}")
    print(
        f"Subtotal {invoice['subtotal']} + tax {invoice['tax']} "
        f"= total {invoice['total']}"
    )

    second = client.post(
        "/invoices",
        json={
            "customer": "Beta Co",
            "items": [{"description": "Logo", "quantity": 1, "unit_price": 150.00}],
        },
    ).get_json()
    print(f"Created invoice with default 10% tax: total {second['total']}")

    paid = client.post(f"/invoices/{invoice['id']}/pay").get_json()
    print(f"Marked paid: {paid['status']}")

    print(f"Unpaid invoices: {client.get('/invoices?status=unpaid').get_json()}")
    print(f"Paid invoices: {client.get('/invoices?status=paid').get_json()}")

    bad = client.post(
        "/invoices",
        json={
            "customer": "Bad Co",
            "items": [{"description": "Widget", "quantity": 0, "unit_price": 10}],
        },
    )
    print(f"Invalid invoice rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
