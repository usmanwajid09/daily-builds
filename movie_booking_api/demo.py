"""Demo: add a movie, open a screening, book seats, cancel and re-book."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from movie_booking_api.app import create_app


def main():
    client = create_app().test_client()

    movie = client.post(
        "/movies",
        json={"title": "GTA VI: The Movie", "duration_minutes": 150, "genre": "Action"},
    ).get_json()
    print(
        f"Created movie #{movie['id']}: {movie['title']} "
        f"({movie['duration_minutes']} min, {movie['genre']})"
    )

    screening = client.post(
        "/screenings",
        json={
            "movie_id": movie["id"],
            "screen": "Screen 1",
            "start_time": "2026-11-22T19:00",
            "rows": 3,
            "seats_per_row": 5,
            "ticket_price": 1200,
        },
    ).get_json()
    print(
        f"Screening #{screening['id']}: {screening['screen']} at "
        f"{screening['start_time']}, {screening['available_seats']} seats available, "
        f"Rs {screening['ticket_price']} per ticket"
    )

    booking = client.post(
        f"/screenings/{screening['id']}/book",
        json={"customer_name": "Usman", "seats": ["A1", "A2", "A3"]},
    ).get_json()
    print(
        f"Booking #{booking['id']}: seats {booking['seats']} for "
        f"{booking['customer_name']} — total Rs {booking['total_price']}"
    )

    taken = client.post(
        f"/screenings/{screening['id']}/book",
        json={"customer_name": "Hamza", "seats": ["A2", "B1"]},
    )
    print(
        f"Booking already-taken seat A2: {taken.status_code} "
        f"({taken.get_json()['error']})"
    )

    state = client.get(f"/screenings/{screening['id']}").get_json()
    print(
        f"After booking: {state['available_seats']} available, "
        f"{state['booked_seats']} booked"
    )

    cancelled = client.post(f"/bookings/{booking['id']}/cancel").get_json()
    print(f"Booking #{cancelled['id']} -> {cancelled['status']}")

    state = client.get(f"/screenings/{screening['id']}").get_json()
    print(
        f"After cancel: {state['available_seats']} available, "
        f"{state['booked_seats']} booked"
    )

    rebooked = client.post(
        f"/screenings/{screening['id']}/book",
        json={"customer_name": "Hamza", "seats": ["A1", "A2"]},
    ).get_json()
    print(
        f"Re-booked seats {rebooked['seats']} for {rebooked['customer_name']} "
        f"— total Rs {rebooked['total_price']}"
    )


if __name__ == "__main__":
    main()
