"""Demo: four journal entries over a week, mood average, tag search."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from journal_api.app import create_app


def main():
    client = create_app().test_client()

    entries = [
        ("2026-11-20", 4, "Shipped the first Docker image.", ["work", "docker"]),
        ("2026-11-22", 3, "Slow day, reviewed pull requests.", ["work"]),
        ("2026-11-24", 5, "Morning run, then a great study session.", ["health", "study"]),
        ("2026-11-26", 4, "Finished the Flask APIs for today.", ["work", "study"]),
    ]
    for day, mood, text, tags in entries:
        created = client.post(
            "/entries", json={"date": day, "mood": mood, "text": text, "tags": tags}
        ).get_json()
        print(f"Entry #{created['id']} {day} mood={mood} tags={tags}")

    stats = client.get(
        "/stats/mood-average",
        query_string={"from": "2026-11-20", "to": "2026-11-26"},
    ).get_json()
    print(f"Average mood 20–26 Nov: {stats['average_mood']} "
          f"over {stats['count']} entries")

    work = client.get("/entries/search", query_string={"tag": "work"}).get_json()
    print(f"Tag 'work' matched {len(work)} entries: "
          f"{[e['date'] for e in work]}")

    week = client.get(
        "/entries", query_string={"from": "2026-11-23", "to": "2026-11-26"}
    ).get_json()
    print(f"Entries 23–26 Nov: {len(week)}")

    bad_mood = client.post(
        "/entries", json={"date": "2026-11-27", "mood": 7, "text": "Too happy?"}
    )
    print(f"Mood 7 rejected: {bad_mood.status_code} {bad_mood.get_json()}")
    bad_date = client.post(
        "/entries", json={"date": "26-11-2026", "mood": 3, "text": "Bad date"}
    )
    print(f"Bad date rejected: {bad_date.status_code} {bad_date.get_json()}")


if __name__ == "__main__":
    main()
