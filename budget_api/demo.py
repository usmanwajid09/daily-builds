"""Demo: set budgets, add expenses and check remaining amounts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from budget_api.app import create_app


def main():
    client = create_app().test_client()

    client.post("/budgets", json={"category": "food", "amount": 20000})
    client.post("/budgets", json={"category": "transport", "amount": 5000})
    client.post("/expenses", json={"category": "food", "amount": 4500, "note": "Groceries"})
    client.post("/expenses", json={"category": "food", "amount": 1200, "note": "Takeaway"})
    client.post("/expenses", json={"category": "transport", "amount": 6000, "note": "Bus pass"})

    budgets = client.get("/budgets").get_json()
    for b in budgets:
        print(
            f"{b['category']}: budget {b['budget']}, spent {b['spent']}, "
            f"remaining {b['remaining']}, over_budget={b['over_budget']}"
        )
    summary = client.get("/summary").get_json()
    print(
        f"Summary: total budget {summary['total_budget']}, "
        f"total spent {summary['total_spent']}, "
        f"total remaining {summary['total_remaining']}, "
        f"over budget: {summary['over_budget']}"
    )


if __name__ == "__main__":
    main()
