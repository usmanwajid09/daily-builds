"""Demo: log a week of sleep, show averages and the streak."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sleep_tracker_api.app import create_app


def main():
    client = create_app().test_client()

    nights = [
        ("2026-11-15", "23:30", "07:15", 4, "Slept well"),
        ("2026-11-16", "00:15", "07:00", 3, None),
        ("2026-11-17", "23:00", "06:45", 5, "Best night in weeks"),
        ("2026-11-18", "23:45", "07:30", 4, None),
        ("2026-11-19", "01:00", "07:30", 2, "Late night, groggy"),
        ("2026-11-20", "23:15", "07:00", 4, None),
        ("2026-11-21", "23:30", "07:45", 5, None),
    ]
    for date, bed, wake, quality, notes in nights:
        payload = {"date": date, "bedtime": bed, "wake_time": wake,
                   "quality": quality}
        if notes:
            payload["notes"] = notes
        night = client.post("/nights", json=payload).get_json()
        print(f"Logged {night['date']}: {night['bedtime']} -> "
              f"{night['wake_time']} = {night['duration_minutes']} min "
              f"({night['duration_hours']} h), quality {night['quality']}/5")

    dup = client.post("/nights", json={"date": "2026-11-21", "bedtime": "23:00",
                                       "wake_time": "07:00", "quality": 3})
    print(f"Duplicate date refused: {dup.status_code} {dup.get_json()}")

    good = client.get("/nights?min_quality=4").get_json()
    print(f"Nights with quality >= 4: {len(good)}")

    stats = client.get("/stats").get_json()
    print(f"Stats: {stats['total_nights']} nights, "
          f"avg {stats['average_duration_minutes']} min "
          f"({stats['average_duration_hours']} h), "
          f"avg quality {stats['average_quality']}/5, "
          f"longest {stats['longest_night_minutes']} min, "
          f"shortest {stats['shortest_night_minutes']} min, "
          f"streak {stats['streak_days']} days")

    bad = client.post("/nights", json={"date": "2026-11-22", "bedtime": "25:00",
                                       "wake_time": "07:00", "quality": 3})
    print(f"Bad bedtime rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
