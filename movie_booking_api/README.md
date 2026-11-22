# movie_booking_api

A Flask API for movie ticket booking: movies, screenings with a seat map,
atomic seat booking, cancellation, and total price calculation.

## Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/movies` | Create a movie (`title`, `duration_minutes`, `genre?`) |
| GET | `/movies` | List movies |
| GET | `/movies/<id>` | Fetch one movie |
| DELETE | `/movies/<id>` | Delete a movie (only while it has no screenings) |
| POST | `/screenings` | Create a screening (`movie_id`, `screen`, `start_time`, `rows`, `seats_per_row`, `ticket_price`) |
| GET | `/screenings` | List screenings; filter by `?movie_id=` |
| GET | `/screenings/<id>` | Fetch a screening with its full seat map |
| POST | `/screenings/<id>/book` | Book seats (`customer_name`, `seats`) |
| GET | `/screenings/<id>/bookings` | Bookings for one screening |
| GET | `/bookings` | All bookings; filter by `?status=active/cancelled` |
| GET | `/bookings/<id>` | Fetch one booking |
| POST | `/bookings/<id>/cancel` | Cancel a booking and free its seats |

Seats are labelled `A1`, `A2`, ... up to `rows` rows (max 26) and
`seats_per_row` seats per row (max 50). Booking is atomic: if any seat is
invalid, missing, or already booked, nothing is booked. `total_price` is
`len(seats) * ticket_price`. Validation errors return 400, missing
movies/screenings/bookings return 404, and booking an already-booked seat
or cancelling an already-cancelled booking returns 409, all as
`{"error": "..."}`.

## How to run

```bash
python movie_booking_api/demo.py
```

Tests:

```bash
python -m pytest movie_booking_api/tests/test_movie_booking_api.py -v
```

## Demo output

```
Created movie #1: GTA VI: The Movie (150 min, Action)
Screening #1: Screen 1 at 2026-11-22T19:00, 15 seats available, Rs 1200.0 per ticket
Booking #1: seats ['A1', 'A2', 'A3'] for Usman — total Rs 3600.0
Booking already-taken seat A2: 409 (seat A2 is already booked)
After booking: 12 available, 3 booked
Booking #1 -> cancelled
After cancel: 15 available, 0 booked
Re-booked seats ['A1', 'A2'] for Hamza — total Rs 2400.0
```
