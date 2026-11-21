"""Demo: track time on two projects, stop timers, show totals and earnings."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time_tracker_api.app import create_app


def main():
    client = create_app().test_client()

    nova = client.post("/projects", json={"name": "NovaStack",
                                          "hourly_rate": 15,
                                          "description": "Client work"}).get_json()
    print(f"Created project {nova['name']!r} at ${nova['hourly_rate']}/h")

    study = client.post("/projects", json={"name": "Study",
                                           "hourly_rate": 0}).get_json()
    print(f"Created project {study['name']!r} at ${study['hourly_rate']}/h")

    client.post(f"/projects/{nova['id']}/entries",
                json={"start_time": "2026-11-21T09:00",
                      "end_time": "2026-11-21T11:30",
                      "description": "Landing page"})
    client.post(f"/projects/{nova['id']}/entries",
                json={"start_time": "2026-11-21T14:00",
                      "end_time": "2026-11-21T15:00",
                      "description": "Bug fixes"})
    client.post(f"/projects/{study['id']}/entries",
                json={"start_time": "2026-11-21T19:00",
                      "end_time": "2026-11-21T21:00",
                      "description": "InfoSec revision"})

    running = client.post(f"/projects/{nova['id']}/entries/start",
                          json={"description": "Evening session",
                                "start_time": "2026-11-21T22:00"}).get_json()
    print(f"Started timer on {nova['name']!r} at {running['start_time']} "
          f"(running={running['running']})")

    second = client.post(f"/projects/{nova['id']}/entries/start", json={})
    print(f"Second timer refused: {second.status_code} {second.get_json()}")

    stopped = client.post(f"/entries/{running['id']}/stop",
                          json={"end_time": "2026-11-21T23:30"}).get_json()
    print(f"Stopped timer: {stopped['duration_minutes']} minutes")

    summary = client.get(f"/projects/{nova['id']}/summary").get_json()
    print(f"{summary['name']}: {summary['total_minutes']} min "
          f"({summary['total_hours']} h) over {summary['completed_entries']} "
          f"entries, earnings ${summary['earnings']}")

    overall = client.get("/summary").get_json()
    print(f"Overall: {overall['total_minutes']} min "
          f"({overall['total_hours']} h), "
          f"${overall['total_earnings']} earned across "
          f"{overall['project_count']} projects")

    bad = client.post(f"/projects/{nova['id']}/entries",
                      json={"start_time": "2026-11-21T10:00",
                            "end_time": "2026-11-21T09:00"})
    print(f"End before start rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
