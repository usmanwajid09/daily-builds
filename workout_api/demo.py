"""Demo: log workouts, show weekly totals and the per-type summary."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workout_api.app import create_app


def main():
    client = create_app().test_client()

    for workout in [
        {"type": "run", "duration_minutes": 30, "calories": 320, "date": "2026-11-16"},
        {"type": "run", "duration_minutes": 25, "calories": 260, "date": "2026-11-17"},
        {"type": "swim", "duration_minutes": 45, "calories": 400, "date": "2026-11-18"},
        {"type": "strength", "duration_minutes": 60, "calories": 350, "date": "2026-11-19"},
        {"type": "run", "duration_minutes": 20, "calories": 200, "date": "2026-11-23"},
    ]:
        created = client.post("/workouts", json=workout).get_json()
        print(f"Logged workout: {created}")

    weekly = client.get("/workouts/weekly?week=2026-11-16").get_json()
    print(f"Week 2026-11-16 to 2026-11-22 totals: {weekly}")

    print(f"Per-type summary: {client.get('/workouts/summary').get_json()}")

    bad = client.post(
        "/workouts",
        json={"type": "run", "duration_minutes": -5, "calories": 100},
    )
    print(f"Negative duration rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
