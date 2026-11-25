"""Demo: build a trip with itinerary days and activities, then show the per-trip cost total."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from travel_planner_api.app import create_app


def main():
    client = create_app().test_client()

    trip = client.post("/trips", json={
        "name": "Northern Areas", "destination": "Hunza",
        "start_date": "2026-12-10", "end_date": "2026-12-12",
    }).get_json()
    print(f"Created trip: {trip}")

    day1 = client.post(f"/trips/{trip['id']}/days", json={
        "date": "2026-12-10", "title": "Arrival and Karimabad",
    }).get_json()
    day2 = client.post(f"/trips/{trip['id']}/days", json={
        "date": "2026-12-11", "title": "Attabad Lake",
    }).get_json()
    print(f"Itinerary days: {day1}, {day2}")

    for day_id, activity in [
        (day1["id"], {"name": "Hotel", "cost": 8000}),
        (day1["id"], {"name": "Dinner", "cost": 2500}),
        (day2["id"], {"name": "Boat ride", "cost": 3000}),
        (day2["id"], {"name": "Jeep hire", "cost": 6000}),
    ]:
        created = client.post(
            f"/trips/{trip['id']}/days/{day_id}/activities", json=activity
        ).get_json()
        print(f"Added activity: {created}")

    cost = client.get(f"/trips/{trip['id']}/cost").get_json()
    print(f"Cost breakdown: {cost}")
    print(f"Headline: Hunza trip total cost is Rs {cost['total_cost']} "
          f"across {len(cost['by_day'])} days")

    bad = client.post(f"/trips/{trip['id']}/days", json={
        "date": "2026-12-20", "title": "Too late",
    })
    print(f"Day outside trip dates rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
