"""Demo: add cars, book one, show the double-booking rejection and the revenue total."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from car_rental_api.app import create_app


def main():
    client = create_app().test_client()

    corolla = client.post("/cars", json={
        "make": "Toyota", "model": "Corolla", "year": 2023, "daily_rate": 50,
    }).get_json()
    civic = client.post("/cars", json={
        "make": "Honda", "model": "Civic", "year": 2024, "daily_rate": 65,
    }).get_json()
    print(f"Added cars: {corolla}, {civic}")

    booking = client.post("/bookings", json={
        "car_id": corolla["id"], "customer_name": "Usman",
        "start_date": "2026-12-01", "end_date": "2026-12-03",
    }).get_json()
    print(f"Booking: {booking}")

    clash = client.post("/bookings", json={
        "car_id": corolla["id"], "customer_name": "Ali",
        "start_date": "2026-12-02", "end_date": "2026-12-04",
    })
    print(f"Double-booking rejected: {clash.status_code} {clash.get_json()}")

    free = client.get("/cars/available?start_date=2026-12-02&end_date=2026-12-04").get_json()
    print(f"Available 2026-12-02 to 2026-12-04: {[c['model'] for c in free]}")

    revenue = client.get(f"/cars/{corolla['id']}/revenue").get_json()
    print(f"Corolla revenue: {revenue}")
    print(f"Headline: 3-day Corolla booking costs ${booking['total_price']}, "
          f"revenue ${revenue['total_revenue']}")


if __name__ == "__main__":
    main()
