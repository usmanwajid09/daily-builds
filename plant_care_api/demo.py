"""Demo: create plants, list the ones due for watering, water one."""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plant_care_api.app import create_app


def _days_ago(days):
    return (date.today() - timedelta(days=days)).isoformat()


def main():
    client = create_app().test_client()

    plants = [
        {
            "name": "Monstera",
            "species": "Monstera deliciosa",
            "watering_interval_days": 7,
            "last_watered": _days_ago(10),
        },
        {
            "name": "Snake Plant",
            "species": "Sansevieria trifasciata",
            "watering_interval_days": 14,
            "last_watered": _days_ago(12),
        },
        {
            "name": "Pothos",
            "species": "Epipremnum aureum",
            "watering_interval_days": 7,
            "last_watered": _days_ago(7),
        },
    ]
    created = []
    for plant in plants:
        response = client.post("/plants", json=plant)
        created.append(response.get_json())
        print(f"Created plant: {response.get_json()}")

    due = client.get("/plants/due").get_json()
    print(f"Due for watering today or overdue: {[p['name'] for p in due]}")

    monstera_id = created[0]["id"]
    watered = client.post(f"/plants/{monstera_id}/water", json={}).get_json()
    print(
        f"Watered {watered['name']}: last_watered={watered['last_watered']}, "
        f"next_watering={watered['next_watering']}"
    )
    log = client.get(f"/plants/{monstera_id}/water-log").get_json()
    print(f"Water log for {watered['name']} (newest first): {log}")

    print(f"All plants: {client.get('/plants').get_json()}")


if __name__ == "__main__":
    main()
