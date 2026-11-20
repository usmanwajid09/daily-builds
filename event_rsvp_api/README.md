# event_rsvp_api

A small Flask events + RSVP API: events with a date and a capacity, and
RSVPs (`yes` / `no` / `maybe`) kept in memory (a fresh store per
`create_app()` call, nothing written to disk). Capacity is enforced: a
`yes` RSVP is refused once the event is full, while `no` and `maybe`
never take a spot.

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/events` | Create an event: `{"title", "date": "YYYY-MM-DD", "capacity", "location"?}` |
| GET | `/events` | List events; `?date=YYYY-MM-DD` filter |
| GET | `/events/<id>` | Fetch one event (includes RSVP counts) |
| PUT | `/events/<id>` | Update title, date, capacity and/or location |
| DELETE | `/events/<id>` | Delete an event and its RSVPs |
| POST | `/events/<id>/rsvp` | RSVP: `{"name", "response": "yes"\|"no"\|"maybe"}` |
| GET | `/events/<id>/attendees` | List RSVPs; `?response=` filter |
| PUT | `/rsvps/<id>` | Change an RSVP's response |
| DELETE | `/rsvps/<id>` | Remove an RSVP |
| GET | `/events/<id>/summary` | Counts by response and spots remaining |

An event is `{"id", "title", "date", "capacity", "location",
"yes_count", "no_count", "maybe_count", "spots_remaining"}`. An RSVP is
`{"id", "event_id", "name", "response"}`. One RSVP per name per event:
RSVPing again with the same name updates the existing RSVP instead of
creating a duplicate. Lowering an event's capacity below the current
number of `yes` RSVPs is refused.

Validation: missing/empty `title` or `name`, a date that is not
`YYYY-MM-DD`, a non-positive `capacity`, or a bad `response` returns
JSON `400`; an unknown event or RSVP returns JSON `404`; a `yes` RSVP
when the event is full returns JSON `409`.

## Run

```bash
python event_rsvp_api/demo.py
python -m pytest tests/test_event_rsvp_api.py -v
```

## Examples

```bash
curl -X POST localhost:5000/events -H 'Content-Type: application/json' \
  -d '{"title": "Board game night", "date": "2026-11-27", "capacity": 10}'

curl -X POST localhost:5000/events/1/rsvp -H 'Content-Type: application/json' \
  -d '{"name": "Ali", "response": "yes"}'

curl localhost:5000/events/1/summary
curl 'localhost:5000/events/1/attendees?response=yes'
```
