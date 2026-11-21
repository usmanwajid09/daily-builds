"""Demo: split a dinner bill, record payments, show the settlement plan."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bill_splitter_api.app import create_app


def main():
    client = create_app().test_client()

    bill = client.post("/bills", json={
        "title": "Dinner at Monal",
        "total": 100,
        "payer": "Usman",
        "participants": ["Usman", "Ali", "Sara", "Bilal"],
    }).get_json()
    print(f"Created bill {bill['title']!r}: total ${bill['total']}, "
          f"payer {bill['payer']}, shares {bill['shares']}")

    uneven = client.post("/bills", json={
        "title": "Groceries",
        "total": 90,
        "payer": "Ali",
        "participants": ["Ali", "Sara"],
        "shares": {"Ali": 60, "Sara": 30},
    }).get_json()
    print(f"Custom shares: {uneven['title']!r} -> {uneven['shares']}")

    client.post(f"/bills/{bill['id']}/payments",
                json={"person": "Ali", "amount": 25})
    client.post(f"/bills/{bill['id']}/payments",
                json={"person": "Sara", "amount": 10})
    after = client.get(f"/bills/{bill['id']}").get_json()
    print(f"After payments on {after['title']!r}: "
          f"outstanding ${after['outstanding']}, settled={after['settled']}")
    for row in after["balances"]:
        print(f"  {row['person']}: share ${row['share']}, paid "
              f"${row['paid']}, owes ${row['balance']}"
              + (" (payer)" if row["is_payer"] else ""))

    plan = client.get(f"/bills/{bill['id']}/settlements").get_json()
    print("Settlement plan:")
    for step in plan:
        print(f"  {step['from']} pays ${step['amount']} to {step['to']}")

    overpay = client.post(f"/bills/{bill['id']}/payments",
                          json={"person": "Sara", "amount": 20})
    print(f"Overpayment refused: {overpay.status_code} {overpay.get_json()}")

    summary = client.get("/summary").get_json()
    print(f"Summary: {summary['total_bills']} bills, "
          f"${summary['total_amount']} total, "
          f"${summary['total_outstanding']} outstanding, "
          f"{summary['settled_bills']} settled")

    bad = client.post("/bills", json={
        "title": "Bad shares", "total": 100, "payer": "Usman",
        "participants": ["Usman", "Ali"],
        "shares": {"Usman": 40, "Ali": 40},
    })
    print(f"Shares not summing to total rejected: {bad.status_code} "
          f"{bad.get_json()}")


if __name__ == "__main__":
    main()
