"""Demo: create a class, fill it, waitlist, cancel, and show the promotion."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from class_booking_api.app import create_app


def main():
    client = create_app().test_client()

    created = client.post("/classes", json={
        "name": "Morning Yoga",
        "instructor": "Sara",
        "date": "2026-11-29",
        "start_time": "08:00",
        "capacity": 2,
    })
    cls = created.get_json()
    print(f"Created class: {cls} (status {created.status_code})")

    for member in ["Ali", "Bilal", "Usman"]:
        resp = client.post(f"/classes/{cls['id']}/book", json={"member_name": member})
        print(f"Booked {member}: {resp.get_json()} (status {resp.status_code})")

    print(f"Bookings: {client.get(f'/classes/{cls['id']}/bookings').get_json()}")
    print(f"Waitlist: {client.get(f'/classes/{cls['id']}/waitlist').get_json()}")
    print(f"Class now: {client.get(f'/classes/{cls['id']}').get_json()}")

    cancelled = client.post(f"/classes/{cls['id']}/cancel", json={"member_name": "Ali"})
    print(f"Cancelled Ali: {cancelled.get_json()} (status {cancelled.status_code})")
    print(f"Bookings after cancel: {client.get(f'/classes/{cls['id']}/bookings').get_json()}")
    print(f"Waitlist after cancel: {client.get(f'/classes/{cls['id']}/waitlist').get_json()}")

    duplicate = client.post(f"/classes/{cls['id']}/book", json={"member_name": "Bilal"})
    print(f"Duplicate booking for Bilal: {duplicate.status_code} {duplicate.get_json()}")


if __name__ == "__main__":
    main()
