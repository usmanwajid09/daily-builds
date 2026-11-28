# class_booking_api

A Flask gym class booking API. Classes have a capacity; members book a spot
while one is free, otherwise they join a waitlist. Cancelling a booked member
promotes the first waitlisted member into the freed spot. All data is kept
in memory — nothing is written to disk.

## Endpoints

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/classes` | Create a class (`name`, optional `instructor`, `date` YYYY-MM-DD, `start_time` HH:MM, `capacity` > 0). Returns 201. |
| GET | `/classes` | List all classes, each with `spots_left`, `booked_count`, and `waitlist_count`. |
| GET | `/classes/<id>` | Fetch one class. Unknown id → 404. |
| DELETE | `/classes/<id>` | Delete a class. Unknown id → 404. |
| POST | `/classes/<id>/book` | Book a spot (`member_name`). 201 with `status: "booked"` while spots remain, or `status: "waitlisted"` with `waitlist_position` when full. Duplicate booking/waitlist → 409. |
| POST | `/classes/<id>/cancel` | Cancel a booking (`member_name`). Promotes the first waitlisted member if the cancelled member was booked; response includes `promoted` (name or `null`). Member not found → 404. |
| GET | `/classes/<id>/bookings` | Booked members in booking order (`member_name`, `status: "booked"`). |
| GET | `/classes/<id>/waitlist` | Waitlisted members in order (`member_name`, `position`). |

All responses are JSON. Errors look like `{"error": "message"}` with status
400 (bad input), 404 (unknown class or member), or 409 (duplicate booking).

## Run the demo

```bash
python class_booking_api/demo.py
```

## Example output

```
Created class: {'booked_count': 0, 'capacity': 2, 'date': '2026-11-29', 'id': 1, 'instructor': 'Sara', 'name': 'Morning Yoga', 'spots_left': 2, 'start_time': '08:00', 'waitlist_count': 0} (status 201)
Booked Ali: {'class_id': 1, 'member_name': 'Ali', 'spots_left': 1, 'status': 'booked'} (status 201)
Booked Bilal: {'class_id': 1, 'member_name': 'Bilal', 'spots_left': 0, 'status': 'booked'} (status 201)
Booked Usman: {'class_id': 1, 'member_name': 'Usman', 'status': 'waitlisted', 'waitlist_position': 1} (status 201)
Bookings: [{'member_name': 'Ali', 'status': 'booked'}, {'member_name': 'Bilal', 'status': 'booked'}]
Waitlist: [{'member_name': 'Usman', 'position': 1}]
Class now: {'booked_count': 2, 'capacity': 2, 'date': '2026-11-29', 'id': 1, 'instructor': 'Sara', 'name': 'Morning Yoga', 'spots_left': 0, 'start_time': '08:00', 'waitlist_count': 1}
Cancelled Ali: {'class_id': 1, 'member_name': 'Ali', 'promoted': 'Usman', 'spots_left': 0, 'status': 'cancelled'} (status 200)
Bookings after cancel: [{'member_name': 'Bilal', 'status': 'booked'}, {'member_name': 'Usman', 'status': 'booked'}]
Waitlist after cancel: []
Duplicate booking for Bilal: 409 {'error': 'member already booked or waitlisted for this class'}
```

## Tests

```bash
python -m pytest tests/test_class_booking_api.py -v
```
