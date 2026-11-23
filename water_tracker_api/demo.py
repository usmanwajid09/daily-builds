"""Demo: log water, check the daily total, and read the streak."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_tracker_api.app import create_app


def main():
    client = create_app().test_client()

    logged = client.post(
        "/intake", json={"date": "2026-11-23", "glasses": 4}
    ).get_json()
    print(f"Logged 4 glasses: {logged}")
    client.post("/intake", json={"date": "2026-11-23", "amount_ml": 500})
    client.post("/intake", json={"date": "2026-11-22", "amount_ml": 2100})

    summary = client.get("/summary/2026-11-23").get_json()
    print(f"Today's summary: {summary}")

    streak = client.get("/streak").get_json()
    print(f"Streak: {streak}")

    weekly = client.get("/weekly?date=2026-11-23").get_json()
    print(
        f"Weekly: total {weekly['total_ml']} ml, "
        f"{weekly['days_met_target']} of 7 days met target, "
        f"avg {weekly['average_ml_per_day']} ml/day"
    )


if __name__ == "__main__":
    main()
