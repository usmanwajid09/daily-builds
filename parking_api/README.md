# parking_api

A Flask API for a parking lot: manage spots, check vehicles in and out,
charge per started hour, and report occupancy and revenue.

## Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/spots` | Create a spot (`spot_number`, `zone?`, `vehicle_type?`, `hourly_rate?`) |
| GET | `/spots` | List spots; filter by `?zone=`, `?vehicle_type=`, `?status=` |
| GET | `/spots/<id>` | Fetch one spot |
| PUT | `/spots/<id>` | Update a spot |
| DELETE | `/spots/<id>` | Delete a spot (only while free and with no sessions) |
| POST | `/spots/<id>/check-in` | Check a vehicle in (`license_plate`, `vehicle_type?`, `started_at?`) |
| POST | `/spots/<id>/check-out` | Check out (`ended_at?`, defaults to now) |
| GET | `/spots/<id>/sessions` | Sessions for one spot |
| GET | `/sessions` | All sessions; filter by `?status=active/completed` |
| GET | `/sessions/<id>` | Fetch one session |
| GET | `/summary` | `total_spots`, `occupied_spots`, `free_spots`, `occupancy_percent`, `total_revenue` |

Vehicle types: `car`, `motorcycle`, `truck`. Spot numbers are unique.
Fees are charged per started hour (`ceil(duration)`, minimum 1 hour) at the
spot's `hourly_rate`. `started_at` / `ended_at` are ISO 8601 timestamps
(useful for tests; both default to now). Validation errors return 400,
missing spots/sessions return 404, and illegal state changes (checking in
to an occupied spot, checking out a free spot, wrong vehicle type, deleting
an occupied spot) return 409, all as `{"error": "..."}`.

## How to run

```bash
python parking_api/demo.py
```

Tests:

```bash
python -m pytest parking_api/tests/test_parking_api.py -v
```

## Demo output

```
Created spot #1: A1 (zone Ground, Rs 100.0/hour) — free
Created spot #2: A2 (zone Ground, Rs 100.0/hour) — free
Created spot #3: B1 (zone Basement, Rs 80.0/hour) — free
Check-in: LEA-123 into spot #1 at 2026-11-22T08:00:00+00:00
Second check-in on occupied spot #1: 409 (spot is already occupied)
Summary: 1/3 occupied (33.33%), revenue Rs 0
Check-out: LEA-123 parked 150.0 minutes, billed 3 hour(s), fee Rs 300.0
Summary: 0/3 occupied (0.0%), revenue Rs 300.0
```
