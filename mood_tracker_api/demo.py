"""Demo: log moods, show duplicate rejection, weekly stats, trend and tags."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

from mood_tracker_api.app import create_app


def main():
    client = create_app().test_client()
    today = date.today()

    for days_ago, mood, note, tags in [
        (3, 2, "Rough start to the week", ["work", "tired"]),
        (2, 3, "Steady day", ["work"]),
        (1, 4, "Good run and early night", ["exercise", "sleep"]),
        (0, 5, "Great day, shipped the feature", ["work", "exercise"]),
    ]:
        entry = {
            "date": (today - timedelta(days=days_ago)).isoformat(),
            "mood": mood,
            "note": note,
            "tags": tags,
        }
        resp = client.post("/entries", json=entry)
        print(f"Created entry ({resp.status_code}): {resp.get_json()}")

    duplicate = {
        "date": today.isoformat(),
        "mood": 1,
        "note": "Trying to log today twice",
        "tags": [],
    }
    resp = client.post("/entries", json=duplicate)
    print(f"Duplicate date rejected ({resp.status_code}): {resp.get_json()}")

    print(f"All entries: {client.get('/entries').get_json()}")
    print(f"Weekly stats: {client.get('/stats/weekly').get_json()}")
    print(f"Trend (14 days): {client.get('/stats/trend').get_json()}")
    print(f"Tag counts: {client.get('/stats/tags').get_json()}")


if __name__ == "__main__":
    main()
