"""Demo: add 3 expenses and print per-category totals."""
from expense_tracker.app import create_app


def main():
    client = create_app().test_client()

    for amount, category, date, description in [
        (12.50, "food", "2026-11-01", "Lunch"),
        (45.00, "transport", "2026-11-02", "Bus pass"),
        (8.75, "food", "2026-11-03", "Coffee"),
    ]:
        added = client.post(
            "/expenses",
            json={
                "amount": amount,
                "category": category,
                "date": date,
                "description": description,
            },
        ).get_json()
        print(f"Added expense: {added}")

    totals = client.get("/totals").get_json()
    print(f"Per-category totals: {totals['by_category']}")
    print(f"Grand total: {totals['total']} across {totals['count']} expenses")

    november = client.get("/totals/2026-11").get_json()
    print(f"November 2026 total: {november['total']}")


if __name__ == "__main__":
    main()
