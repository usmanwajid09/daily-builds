"""Demo: add rooms, book one, check availability, then cancel."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hotel_booking_api.app import create_app


def main():
    client = create_app().test_client()

    room101 = client.post(
        "/rooms",
        json={"number": "101", "type": "double", "nightly_rate": 120, "capacity": 2},
    ).get_json()
    room102 = client.post(
        "/rooms",
        json={"number": "102", "type": "suite", "nightly_rate": 250, "capacity": 4},
    ).get_json()
    booking = client.post(
        "/bookings",
        json={
            "room_id": room101["id"],
            "guest_name": "Alice",
            "check_in": "2026-12-10",
            "check_out": "2026-12-13",
            "guests": 2,
        },
    ).get_json()
    print(
        f"Booked room {room101['number']} for {booking['nights']} nights: "
        f"total ${booking['total']}"
    )
    busy = client.get(
        f"/rooms/{room101['id']}/availability",
        query_string={"check_in": "2026-12-11", "check_out": "2026-12-12"},
    ).get_json()
    print(f"Room 101 available 11–12 Dec: {busy['available']}")
    free = client.get(
        "/availability",
        query_string={"check_in": "2026-12-11", "check_out": "2026-12-12"},
    ).get_json()
    print(
        f"Rooms free 11–12 Dec: {[r['number'] for r in free]} "
        f"(suite rate ${room102['nightly_rate']}/night)"
    )
    cancelled = client.post(f"/bookings/{booking['id']}/cancel").get_json()
    free_again = client.get(
        f"/rooms/{room101['id']}/availability",
        query_string={"check_in": "2026-12-11", "check_out": "2026-12-12"},
    ).get_json()
    print(
        f"Booking {cancelled['id']} status: {cancelled['status']}; "
        f"room 101 available again: {free_again['available']}"
    )


if __name__ == "__main__":
    main()
