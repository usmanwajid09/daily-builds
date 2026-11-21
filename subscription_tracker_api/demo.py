"""Demo: add subscriptions, show totals and upcoming renewals."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subscription_tracker_api.app import create_app


def main():
    client = create_app().test_client()

    subs = [
        {"name": "Netflix", "cost": 9.99, "billing_cycle": "monthly",
         "renewal_date": "2026-12-01", "category": "streaming"},
        {"name": "Spotify", "cost": 5.99, "billing_cycle": "monthly",
         "renewal_date": "2026-11-25", "category": "music"},
        {"name": "GitHub Pro", "cost": 48, "billing_cycle": "yearly",
         "renewal_date": "2027-03-10", "category": "dev tools"},
        {"name": "Gym", "cost": 25, "billing_cycle": "monthly",
         "renewal_date": "2026-11-28", "category": "health"},
    ]
    for payload in subs:
        sub = client.post("/subscriptions", json=payload).get_json()
        print(f"Added {sub['name']}: ${sub['cost']}/{sub['billing_cycle']} "
              f"(monthly ${sub['monthly_cost']}, yearly ${sub['yearly_cost']})")

    dup = client.post("/subscriptions", json=subs[0])
    print(f"Duplicate name refused: {dup.status_code} {dup.get_json()}")

    totals = client.get("/totals").get_json()
    print(f"Totals: {totals['count']} subscriptions, "
          f"${totals['monthly_total']}/month, ${totals['yearly_total']}/year")
    print(f"Monthly by category: {totals['by_category_monthly']}")

    upcoming = client.get("/upcoming?as_of=2026-11-21&days=10").get_json()
    print(f"Renewing in the next 10 days from 2026-11-21 ({len(upcoming)}):")
    for row in upcoming:
        print(f"  {row['name']}: {row['next_renewal']} "
              f"(${row['cost']}, in {row['days_until_renewal']} days)")

    bad = client.post("/subscriptions",
                      json={"name": "Bad", "cost": -5,
                            "billing_cycle": "monthly",
                            "renewal_date": "2026-12-01"})
    print(f"Negative cost rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
