# calendar_api

Calendar events API: create, read, update and delete events, list
upcoming events, and filter by date. In-memory only — nothing is
written to disk. Times use `YYYY-MM-DDTHH:MM`.

## Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST | `/events` | Create an event (`title`, `start`, `end`, optional `location`/`description`) |
| GET | `/events` | List all events, sorted by start (`?date=YYYY-MM-DD` filters to one day) |
| GET | `/events/upcoming` | Events starting at or after `now` (`?now=YYYY-MM-DDTHH:MM`, defaults to current time) |
| GET | `/events/<id>` | Fetch one event |
| PUT | `/events/<id>` | Replace an event |
| DELETE | `/events/<id>` | Delete an event |

An end time that is not after the start, or a malformed timestamp,
returns `400`; unknown event ids return `404`.

## Run

```bash
python -m pytest tests/test_calendar_api.py -q
python calendar_api/demo.py
```

## Demo output

```
Created event 1: Team standup 2026-11-17T09:00–2026-11-17T09:30 -> 201
Created event 2: Project demo 2026-11-17T14:00–2026-11-17T15:00 -> 201
Created event 3: Dentist 2026-11-18T11:00–2026-11-18T11:45 -> 201
Created event 4: Old meeting 2026-11-16T10:00–2026-11-16T11:00 -> 201
All events: 4 (sorted by start)
Upcoming from 2026-11-17: 3 -> ['Team standup', 'Project demo', 'Dentist']
Events on 2026-11-17: 2 -> ['Team standup', 'Project demo']
End before start -> 400: end must be after start
Unknown event -> 404: event 99 not found
```
