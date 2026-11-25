# Car Rental API

A tiny REST car rental API built with Flask. Cars and bookings live in
an in-memory store (fresh on every app start — nothing is written to
disk), so it is ideal for learning REST patterns and for testing with
Flask's test client.

## Why

A car rental is a small CRUD app with one tricky rule: a car cannot be
booked twice for overlapping dates. The booking price is computed from
the number of rental days (inclusive) times the car's daily rate, and
the availability endpoint answers "which cars are free for this date
range?".

## Usage

```bash
pip install -r requirements.txt
python car_rental_api/demo.py
```

The demo adds two cars, books the Corolla for three days, shows an
overlapping booking being rejected with 409, lists the cars still free
for those dates, and prints the Corolla's revenue.

To run the app for real:

```bash
flask --app "car_rental_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/cars` | Add a car: `{"make", "model", "year", "daily_rate"}` |
| GET | `/cars` | List all cars |
| GET | `/cars/<id>` | Fetch one car |
| GET | `/cars/available` | Free cars for a range: `?start_date=&end_date=` (YYYY-MM-DD) |
| POST | `/bookings` | Book a car: `{"car_id", "customer_name", "start_date", "end_date"}` |
| GET | `/bookings` | List bookings; `?car_id=` and `?status=` filters |
| GET | `/bookings/<id>` | Fetch one booking |
| POST | `/bookings/<id>/cancel` | Cancel a booking (frees the dates) |
| GET | `/cars/<id>/revenue` | Booking count and total revenue for one car |

A booking looks like
`{"id": 1, "car_id": 1, "customer_name": "Usman", "start_date": "2026-12-01", "end_date": "2026-12-03", "days": 3, "total_price": 150, "status": "booked"}`.
Validation errors return `400` with `{"error": "..."}`; unknown ids
return `404`; an overlapping booking returns `409`.

## Run tests

```bash
python -m pytest tests/test_car_rental_api.py -v
```
