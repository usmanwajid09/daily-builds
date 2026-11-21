"""Demo: create pomodoro sessions, run one to completion, show stats."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pomodoro_api.app import create_app


def main():
    client = create_app().test_client()

    focus = client.post(
        "/sessions", json={"task": "Write report", "type": "focus"}
    ).get_json()
    print(f"Created focus session #{focus['id']}: {focus['task']!r} "
          f"({focus['duration_minutes']} min, status {focus['status']})")

    short_break = client.post(
        "/sessions", json={"task": "Stretch", "type": "short_break"}
    ).get_json()
    print(f"Created break session #{short_break['id']}: "
          f"{short_break['task']!r} "
          f"({short_break['duration_minutes']} min)")

    custom = client.post(
        "/sessions",
        json={"task": "Deep work", "type": "focus", "duration_minutes": 50},
    ).get_json()
    print(f"Created custom session #{custom['id']}: "
          f"{custom['duration_minutes']} min focus")

    started = client.post(f"/sessions/{focus['id']}/start").get_json()
    print(f"Started session #{started['id']}: status {started['status']}, "
          f"started_at {started['started_at']}")

    done = client.post(f"/sessions/{focus['id']}/complete").get_json()
    print(f"Completed session #{done['id']}: status {done['status']}, "
          f"completed_at {done['completed_at']}")

    again = client.post(f"/sessions/{focus['id']}/start")
    print(f"Restarting a completed session rejected: {again.status_code} "
          f"{again.get_json()}")

    cancelled = client.post(f"/sessions/{short_break['id']}/cancel").get_json()
    print(f"Cancelled session #{cancelled['id']}: "
          f"status {cancelled['status']}")

    stats = client.get("/stats").get_json()
    print(f"Stats: {stats['total_sessions']} sessions, "
          f"{stats['completed_sessions']} completed, "
          f"{stats['focus_minutes_completed']} focus minutes completed, "
          f"by status {stats['by_status']}")
    today = client.get("/stats/daily").get_json()
    print(f"Today ({today['date']}): {today['completed_sessions']} completed, "
          f"{today['focus_minutes_completed']} focus minutes")


if __name__ == "__main__":
    main()
