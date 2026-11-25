"""Demo: create an event, sign volunteers up to capacity, log hours, and show per-volunteer totals."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volunteer_api.app import create_app


def main():
    client = create_app().test_client()

    event = client.post("/events", json={
        "name": "Food Drive", "date": "2026-11-29", "capacity": 2,
        "location": "Lahore",
    }).get_json()
    print(f"Created event: {event}")

    for name in ["Usman", "Ali"]:
        signup = client.post(
            f"/events/{event['id']}/signups", json={"volunteer_name": name}
        ).get_json()
        print(f"Signup: {signup}")

    full = client.post(
        f"/events/{event['id']}/signups", json={"volunteer_name": "Sara"}
    )
    print(f"Over-capacity signup rejected: {full.status_code} {full.get_json()}")

    signups = client.get(f"/events/{event['id']}/signups").get_json()
    hours = client.post(
        f"/events/{event['id']}/signups/{signups[0]['id']}/hours",
        json={"hours": 4},
    ).get_json()
    print(f"Hours logged: {hours}")

    totals = client.get("/volunteers?name=Usman").get_json()
    print(f"Usman's hours: {totals}")
    print(f"Headline: Usman has logged {totals['total_hours']} volunteer hours, "
          f"event has {event['spots_remaining'] - 2} spots remaining after 2 signups")


if __name__ == "__main__":
    main()
