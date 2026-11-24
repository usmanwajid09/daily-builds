# hotel_booking_api

A hotel-booking REST API built with Flask and an in-memory store.

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/rooms` | Add a room (`number`, `type`, `nightly_rate`, `capacity`; numbers unique) |
| GET | `/rooms` | List all rooms |
| GET | `/rooms/<id>` | Fetch one room |
| GET | `/rooms/<id>/availability` | Is the room free for `?check_in=YYYY-MM-DD&check_out=YYYY-MM-DD` |
| GET | `/availability` | Rooms free for a date range |
| POST | `/bookings` | Book a room (`room_id`, `guest_name`, `check_in`, `check_out`, `guests`) |
| GET | `/bookings` | List bookings (`?room_id=...`, `?status=confirmed\|cancelled`) |
| GET | `/bookings/<id>` | Fetch one booking |
| POST | `/bookings/<id>/cancel` | Cancel a booking |

## Rules

- A booking's total is `nights * nightly_rate`, computed by the server.
- `check_out` must be after `check_in`; guest count must not exceed
  the room's capacity.
- A booking overlapping a confirmed booking for the same room returns
  `409 Conflict`. Cancelled bookings free the room again.

## Run the demo

```bash
python -m hotel_booking_api.demo
```

## Run the tests

```bash
python -m pytest tests/test_hotel_booking_api.py -q
```
