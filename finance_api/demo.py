"""Demo: record income/expenses, show balance, categories, monthly summary, budget alerts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finance_api.app import create_app


def main():
    client = create_app().test_client()

    salary = client.post(
        "/transactions",
        json={"description": "Freelance payment", "amount": 50000,
              "type": "income", "category": "Work", "date": "2026-11-01"},
    ).get_json()
    print(f"Income: {salary['description']} +{salary['amount']} "
          f"({salary['category']})")

    for desc, amount, category, day in [
        ("Groceries", 8000, "Food", "2026-11-05"),
        ("Bus fare", 1500, "Transport", "2026-11-06"),
        ("Restaurant", 4500, "Food", "2026-11-10"),
    ]:
        txn = client.post(
            "/transactions",
            json={"description": desc, "amount": amount, "type": "expense",
                  "category": category, "date": day},
        ).get_json()
        print(f"Expense: {txn['description']} -{txn['amount']} "
              f"({txn['category']})")

    balance = client.get("/balance").get_json()
    print(f"Balance: income {balance['total_income']}, "
          f"expenses {balance['total_expense']}, "
          f"balance {balance['balance']}")

    categories = client.get("/categories").get_json()
    print(f"By category: {categories}")

    summary = client.get("/summary", query_string={"month": "2026-11"}).get_json()
    print(f"November summary: {summary['transaction_count']} transactions, "
          f"income {summary['total_income']}, expenses "
          f"{summary['total_expense']}, balance {summary['balance']}")

    budget = client.post("/budgets", json={"category": "Food", "limit": 10000}).get_json()
    print(f"Budget set: {budget['category']} limit {budget['limit']}")
    client.post("/budgets", json={"category": "Transport", "limit": 2000})

    alerts = client.get("/budgets/alerts").get_json()
    for alert in alerts:
        print(f"Budget {alert['category']}: spent {alert['spent']} of "
              f"{alert['limit']} ({alert['pct_used']}%) -> {alert['status']}")

    bad = client.post(
        "/transactions",
        json={"description": "Bad", "amount": -50, "type": "expense",
              "category": "Food", "date": "2026-11-11"},
    )
    print(f"Negative amount rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
