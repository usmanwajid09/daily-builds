"""Demo: create habits, mark days done, and watch streaks grow."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from habit_tracker.app import create_app


def main():
    client = create_app().test_client()

    created = client.post(
        "/habits", json={"name": "Read 20 pages", "description": "Before bed"}
    ).get_json()
    print(f"Created habit: {created}")

    for day in ("2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14"):
        resp = client.post(f"/habits/{created['id']}/done", json={"date": day})
        print(f"Marked {day} done -> streak {resp.get_json()['streak']}")

    # A gap day breaks the streak: 2026-11-10 was skipped, so the run
    # of 4 consecutive days is the streak, not 5.
    client.post("/habits", json={"name": "Morning run"})
    listed = client.get("/habits").get_json()
    print(f"Listed {len(listed)} habits:")
    for habit in listed:
        print(
            f"  #{habit['id']} {habit['name']}: "
            f"streak={habit['streak']}, total={habit['total']}"
        )

    bad = client.post("/habits", json={"name": "   "})
    print(f"Blank name rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
