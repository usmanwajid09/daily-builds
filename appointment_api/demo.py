"""Demo: create services, book appointments, block a double-booking."""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from appointment_api.app import create_app


def main():
    client = create_app().test_client()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    day_after = (date.today() + timedelta(days=2)).isoformat()

    haircut = client.post(
        "/services",
        json={"name": "Haircut", "duration_minutes": 30, "price": 1500},
    ).get_json()
    print(f"Created service #{haircut['id']}: {haircut['name']} "
          f"({haircut['duration_minutes']} min, Rs {haircut['price']})")

    dental = client.post(
        "/services", json={"name": "Dental checkup", "duration_minutes": 45},
    ).get_json()
    print(f"Created service #{dental['id']}: {dental['name']} "
          f"({dental['duration_minutes']} min)")

    first = client.post(
        "/appointments",
        json={
            "service_id": haircut["id"],
            "customer_name": "Usman",
            "date": tomorrow,
            "start_time": "10:00",
        },
    ).get_json()
    print(f"Booked appointment #{first['id']}: {first['customer_name']}, "
          f"{first['date']} {first['start_time']}-{first['end_time']}")

    clash = client.post(
        "/appointments",
        json={
            "service_id": haircut["id"],
            "customer_name": "Ali",
            "date": tomorrow,
            "start_time": "10:15",
        },
    )
    print(f"Overlapping 10:15 booking rejected: {clash.status_code} "
          f"{clash.get_json()}")

    adjacent = client.post(
        "/appointments",
        json={
            "service_id": haircut["id"],
            "customer_name": "Sara",
            "date": tomorrow,
            "start_time": "10:30",
        },
    ).get_json()
    print(f"Back-to-back 10:30 booking accepted: appointment "
          f"#{adjacent['id']} ({adjacent['start_time']}-{adjacent['end_time']})")

    other_service = client.post(
        "/appointments",
        json={
            "service_id": dental["id"],
            "customer_name": "Bilal",
            "date": tomorrow,
            "start_time": "10:15",
        },
    )
    print(f"Same time, different service accepted: {other_service.status_code}")

    cancelled = client.post(f"/appointments/{first['id']}/cancel").get_json()
    print(f"Cancelled appointment #{first['id']}: status {cancelled['status']}")

    rebooked = client.post(
        "/appointments",
        json={
            "service_id": haircut["id"],
            "customer_name": "Ali",
            "date": tomorrow,
            "start_time": "10:00",
        },
    )
    print(f"Rebooking the freed 10:00 slot: {rebooked.status_code}")

    client.post(
        "/appointments",
        json={
            "service_id": haircut["id"],
            "customer_name": "Hassan",
            "date": day_after,
            "start_time": "09:00",
        },
    )
    upcoming = client.get("/appointments/upcoming").get_json()
    print(f"Upcoming appointments: {len(upcoming)}")
    for a in upcoming:
        print(f"  #{a['id']} {a['date']} {a['start_time']}-{a['end_time']} "
              f"({a['customer_name']}, service {a['service_id']})")


if __name__ == "__main__":
    main()
