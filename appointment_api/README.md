# appointment_api

A small Flask appointment booking API: services with durations, slot
booking with double-booking prevention, cancel and an upcoming list,
kept in memory (a fresh store per `create_app()` call, nothing written
to disk).

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/services` | Create a service: `{"name", "duration_minutes", "price"?}` |
| GET | `/services` | List services |
| GET | `/services/<id>` | Fetch one service |
| POST | `/appointments` | Book: `{"service_id", "customer_name", "date": "YYYY-MM-DD", "start_time": "HH:MM"}` |
| GET | `/appointments` | List appointments; `?date=`, `?service_id=`, `?status=` filters |
| GET | `/appointments/upcoming` | Booked appointments from today onwards, earliest first |
| GET | `/appointments/<id>` | Fetch one appointment |
| POST | `/appointments/<id>/cancel` | Cancel a booked appointment |
| DELETE | `/appointments/<id>` | Delete an appointment |

The end time is computed from the service's duration. A booking whose
time range overlaps another **booked** appointment for the same service
on the same date is rejected with JSON `409`; back-to-back slots are
fine, different services at the same time are fine, and cancelling
frees the slot for rebooking.

Validation: bad `service_id`, missing/empty `customer_name`, malformed
`date`/`start_time`, or a non-positive `duration_minutes` returns JSON
`400`; an unknown service or appointment returns JSON `404`.

## Run

```bash
python appointment_api/demo.py
python -m pytest tests/test_appointment_api.py -v
```
